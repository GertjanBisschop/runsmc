import itertools
import math
import numpy as np
import numba
import tskit


@numba.njit(cache=True)
def binary_search(array, value, low_ptr, high_ptr):
    # Repeat until the pointers low and high meet each other
    while low_ptr <= high_ptr:
        mid = low_ptr + (high_ptr - low_ptr) // 2
        if array[mid] == value:
            return mid
        elif array[mid] < value:
            low_ptr = mid + 1
        else:
            high_ptr = mid - 1

    return -1


@numba.njit(cache=True)
def update_counts_ascending(intervals, counts, start_ptr, stop_value, increment):
    assert start_ptr < intervals.size
    assert intervals[start_ptr] < stop_value
    i = start_ptr
    while intervals[i] < stop_value:
        counts[i] += increment
        i += 1
        if i >= intervals.size:
            break

    j = i
    if increment < 0:
        if j < intervals.size:
            if counts[j] == abs(increment):
                while j < intervals.size:
                    counts[j] = 0
                    j += 1
    return i


@numba.njit(cache=True)
def update_counts_descending(intervals, counts, start_ptr, stop_value, increment):
    assert start_ptr > 0
    assert intervals[start_ptr] > stop_value
    i = start_ptr
    if increment < 0:
        if counts[i] == abs(increment):
            while i < intervals.size:
                counts[i] = 0
                i += 1

    i = start_ptr - 1
    while intervals[i] >= stop_value:
        counts[i] += increment
        i -= 1
        if i < 0:
            break
    return i


def counts_from_array(intervals, counts, stop_value, start_ptr):
    assert start_ptr > 0
    i = start_ptr - 1
    sub_intervals = [intervals[start_ptr]]
    sub_counts = []
    while intervals[i] >= stop_value:
        new_count = counts[i] - 1
        sub_intervals.append(intervals[i])
        sub_counts.append(new_count)
        i -= 1
        if i < 0:
            break

    return np.array(sub_counts)[::-1], np.array(sub_intervals)[::-1]


def log_depth(
    min_parent_time,
    left_count,
    intervals,
    rec_rate,
    coal_rate,
    rec_event,
):
    ret = 0
    interval_lengths = intervals[1:] - intervals[:-1]
    assert len(interval_lengths) == len(left_count)
    # area under the left_count non-increasing step function
    area = left_count * interval_lengths
    cum_area = np.sum(area)

    def f(f0, f1):
        n1 = rec_rate * (f1 - f0)
        n2 = coal_rate / (rec_rate - coal_rate * f1)
        n3 = coal_rate / (rec_rate - coal_rate * f0)
        return n1 * n2 * n3

    if not rec_event:
        # if no recombination event expression simplifies to
        ret = coal_rate * np.exp(-coal_rate * cum_area)
    else:
        denoms = rec_rate - coal_rate * left_count
        if np.any(denoms == 0):
            raise ValueError("denom is 0")
        else:
            t1 = intervals[0]
            ret = (
                coal_rate
                / (rec_rate - coal_rate * left_count[0])
                * rec_rate
                * np.exp(-rec_rate * t1 - coal_rate * cum_area)
            )
            for i in range(1, len(intervals)):
                t0 = t1
                t1 = min(min_parent_time, intervals[i])
                cum_area -= left_count[i - 1] * (t1 - t0)
                if t1 == min_parent_time:
                    break

                ret += f(left_count[i - 1], left_count[i]) * np.exp(
                    -rec_rate * t1 - coal_rate * cum_area
                )

            ret -= (
                coal_rate
                / (rec_rate - coal_rate * left_count[i - 1])
                * rec_rate
                * np.exp(-rec_rate * t1 - coal_rate * cum_area)
            )

    assert ret > 0, "About to take log of non-positive value."
    return np.log(ret)


@numba.njit(cache=True)
def log_span(r, parent_time, child_time, left, right):
    ret = 0
    if r > 0:
        assert right > left
        assert parent_time > child_time
        ret = -r * (parent_time - child_time) * (right - left)

    return ret


def log_likelihood(ts, rec_rate, population_size):
    # here we can no longer account for the fact that past the
    # first mrca we might observe discontinuous edges (for the
    # same parent child pair)
    ret = 0
    coal_rate = 1 / (2 * population_size)
    coalescent_nodes_array = np.zeros(ts.num_nodes, dtype=np.int64)
    num_children_array = np.zeros(ts.num_nodes, dtype=np.int64)
    I = ts.nodes_time[ts.num_samples - 1 :]
    num_intervals = I.size
    C = np.zeros_like(I, dtype=np.int64)

    for _, edges_out, edges_in in ts.edge_diffs():
        last_parent_array = -np.ones(ts.num_nodes, dtype=np.int64)
        last_ptr = num_intervals
        for edge in edges_out:
            t_child = ts.nodes_time[edge.child]
            t_parent = ts.nodes_time[edge.parent]
            last_ptr = binary_search(I, t_parent, 0, last_ptr)
            assert last_ptr > -1
            stop_ptr = update_counts_descending(I, C, last_ptr, t_child, -1)
            num_children_array[edge.parent] -= 1
            last_parent_array[edge.child] = edge.parent

        # once edges from previous tree are out
        # A contains all the counts for edges that start off to the left
        # of x and overlap with x
        last_ptr = 0
        for edge in edges_in:
            # new edges coming in are those that start at position x
            t_child = ts.nodes_time[edge.child]
            t_parent = ts.nodes_time[edge.parent]
            last_ptr = binary_search(I, t_parent, last_ptr, num_intervals)
            assert last_ptr > -1
            stop_ptr = update_counts_descending(I, C, last_ptr, t_child, +1)
            num_children_array[edge.parent] += 1
            if num_children_array[edge.parent] >= 2:
                coalescent_nodes_array[edge.parent] = 1

            # compute likelihoods given the counts in avl tree
            # note that we impose a strict ordering on the edges
            # we add to the avl tree and thus on the number of
            # lineages each lineage can coalesce with
            rec_event = False
            left_parent_time = math.inf
            last_parent = last_parent_array[edge.child]
            if last_parent != -1:
                left_parent_time = ts.nodes_time[last_parent]
                if edge.parent != last_parent:
                    rec_event = True

            min_parent_time = min(left_parent_time, t_parent)
            left_count, intervals = counts_from_array(I, C, t_child, last_ptr)
            # print(edge.parent, edge.child)
            # print(left_count)
            # print(intervals)
            # print('-------------')
            ret += log_depth(
                min_parent_time,
                left_count,
                intervals,
                rec_rate,
                coal_rate,
                rec_event,
            )
            ret += log_span(
                rec_rate,
                t_parent,
                t_child,
                edge.left,
                edge.right,
            )

    num_edges = ts.num_edges
    num_coal_events = np.sum(coalescent_nodes_array)
    ret -= (num_edges - num_coal_events) * np.log(coal_rate)

    return ret
