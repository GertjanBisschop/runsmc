import itertools
import math
import numpy as np
import numba
import tskit


spec = [
    ("num_edges", numba.int64),
    ("sequence_length", numba.float64),
    ("edges_left", numba.float64[:]),
    ("edges_right", numba.float64[:]),
    ("edge_insertion_order", numba.int32[:]),
    ("edge_removal_order", numba.int32[:]),
    ("edge_insertion_index", numba.int64),
    ("edge_removal_index", numba.int64),
    ("interval", numba.float64[:]),
    ("in_range", numba.int64[:]),
    ("out_range", numba.int64[:]),
]


@numba.experimental.jitclass(spec)
class TreePosition:
    def __init__(
        self,
        num_edges,
        sequence_length,
        edges_left,
        edges_right,
        edge_insertion_order,
        edge_removal_order,
    ):
        self.num_edges = num_edges
        self.sequence_length = sequence_length
        self.edges_left = edges_left
        self.edges_right = edges_right
        self.edge_insertion_order = edge_insertion_order
        self.edge_removal_order = edge_removal_order
        self.edge_insertion_index = 0
        self.edge_removal_index = 0
        self.interval = np.zeros(2)
        self.in_range = np.zeros(2, dtype=np.int64)
        self.out_range = np.zeros(2, dtype=np.int64)

    def next(self):
        left = self.interval[1]
        j = self.in_range[1]
        k = self.out_range[1]
        self.in_range[0] = j
        self.out_range[0] = k
        M = self.num_edges
        edges_left = self.edges_left
        edges_right = self.edges_right
        out_order = self.edge_removal_order
        in_order = self.edge_insertion_order

        while k < M and edges_right[out_order[k]] == left:
            k += 1
        while j < M and edges_left[in_order[j]] == left:
            j += 1
        self.out_range[1] = k
        self.in_range[1] = j

        right = self.sequence_length
        if j < M:
            right = min(right, edges_left[in_order[j]])
        if k < M:
            right = min(right, edges_right[out_order[k]])
        self.interval[:] = [left, right]
        return j < M or left < self.sequence_length


# Helper function to make it easier to communicate with the numba class
def alloc_tree_position(ts):
    return TreePosition(
        num_edges=ts.num_edges,
        sequence_length=ts.sequence_length,
        edges_left=ts.edges_left,
        edges_right=ts.edges_right,
        edge_insertion_order=ts.indexes_edge_insertion_order,
        edge_removal_order=ts.indexes_edge_removal_order,
    )


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

    return i


@numba.njit(cache=True)
def update_counts_descending(intervals, counts, start_ptr, stop_value, increment):
    assert start_ptr > 0
    assert intervals[start_ptr] > stop_value

    i = start_ptr - 1
    while intervals[i] >= stop_value:
        counts[i] += increment
        i -= 1
        if i < 0:
            break
    return i


@numba.njit(cache=True)
def update_counts_descending_ptr(counts, start_ptr, stop_ptr, increment):
    assert start_ptr > 0

    i = start_ptr - 1
    while i >= stop_ptr:
        counts[i] += increment
        i -= 1
        if i < 0:
            break
    return i


@numba.njit(cache=True)
def counts_from_array(intervals, counts, stop_value, start_ptr):
    assert start_ptr > 0
    i = start_ptr - 1
    sub_intervals = [intervals[start_ptr]]
    sub_counts = []
    while intervals[i] >= stop_value:
        new_count = counts[i]
        sub_intervals.append(intervals[i])
        sub_counts.append(new_count)
        i -= 1
        if i < 0:
            break

    return np.array(sub_counts)[::-1], np.array(sub_intervals)[::-1]


@numba.njit(cache=True)
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
        n1 = coal_rate / (rec_rate - coal_rate * f1)
        n2 = coal_rate / (rec_rate - coal_rate * f0)
        n3 = (f1 - f0) * (rec_rate / coal_rate)
        return n1 * n2 * n3

    if not rec_event:
        # if no recombination event expression simplifies to
        ret = np.exp(-coal_rate * cum_area)
    else:
        denoms = rec_rate - coal_rate * left_count
        if np.any(denoms == 0):
            raise ValueError("denom is 0")
        else:
            t1 = intervals[0]
            ret = (
                coal_rate
                / (rec_rate - coal_rate * left_count[0])
                * (rec_rate / coal_rate)
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
                * (rec_rate / coal_rate)
                * np.exp(-rec_rate * t1 - coal_rate * cum_area)
            )

    assert ret > 0, "About to take log of non-positive value."
    return np.log(ret)


@numba.njit(cache=True)
def log_depth_descending(
    left_count,
    intervals,
    min_parent_time,
    child_time,
    parent_ptr,
    rec_rate,
    coal_rate,
    rec_event,
):
    ret = 0
    cum_area = 0
    i = parent_ptr

    def f(f0, f1):
        n1 = coal_rate / (rec_rate - coal_rate * f1)
        n2 = coal_rate / (rec_rate - coal_rate * f0)
        n3 = (f1 - f0) * (rec_rate / coal_rate)
        return n1 * n2 * n3

    if not rec_event:
        stop_time = child_time
    else:
        stop_time = min_parent_time

    while intervals[i] > stop_time:
        span = intervals[i] - intervals[i - 1]
        cum_area += span * left_count[i - 1]
        i -= 1
        if i < 1:
            break

    if not rec_event:
        ret = np.exp(-coal_rate * cum_area)
    else:
        assert min_parent_time == intervals[i]
        temp = -(
            coal_rate
            / (rec_rate - coal_rate * left_count[i - 1])
            * (rec_rate / coal_rate)
            * np.exp(-rec_rate * intervals[i] - coal_rate * cum_area)
        )
        while i > 0:
            cum_area += (intervals[i] - intervals[i - 1]) * left_count[i - 1]
            i -= 1
            if intervals[i] == child_time:
                break
            ret += f(left_count[i - 1], left_count[i]) * np.exp(
                -rec_rate * intervals[i] - coal_rate * cum_area
            )

        assert intervals[i] == child_time
        ret += (
            coal_rate
            / (rec_rate - coal_rate * left_count[i])
            * (rec_rate / coal_rate)
            * np.exp(-rec_rate * intervals[i] - coal_rate * cum_area)
        )

        ret += temp

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

        last_ptr = 0
        for edge in edges_in:
            # new edges coming in are those that start at position x
            t_child = ts.nodes_time[edge.child]
            t_parent = ts.nodes_time[edge.parent]
            last_ptr = binary_search(I, t_parent, last_ptr, num_intervals)
            assert last_ptr > -1

            rec_event = False
            left_parent_time = math.inf
            last_parent = last_parent_array[edge.child]
            if last_parent != -1:
                left_parent_time = ts.nodes_time[last_parent]
                if edge.parent != last_parent:
                    rec_event = True

            min_parent_time = min(left_parent_time, t_parent)
            left_count, intervals = counts_from_array(I, C, t_child, last_ptr)
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
            stop_ptr = update_counts_descending(I, C, last_ptr, t_child, +1)
            num_children_array[edge.parent] += 1
            if num_children_array[edge.parent] >= 2:
                coalescent_nodes_array[edge.parent] = 1

    num_coal_events = np.sum(coalescent_nodes_array)
    ret += num_coal_events * np.log(coal_rate)

    return ret


def log_likelihood_descending(ts, rec_rate, population_size):
    # here we can no longer account for the fact that past the
    # first mrca we might observe discontinuous edges (for the
    # same parent child pair)
    ret = 0
    coal_rate = 1 / (2 * population_size)
    coalescent_nodes_array = np.zeros(ts.num_nodes, dtype=np.int64)
    num_children_array = np.zeros(ts.num_nodes, dtype=np.int64)
    I, node_map = np.unique(ts.nodes_time, return_inverse=True)
    num_intervals = I.size
    C = np.zeros_like(I, dtype=np.int64)

    for _, edges_out, edges_in in ts.edge_diffs():
        last_parent_array = -np.ones(ts.num_nodes, dtype=np.int64)

        for edge in edges_out:
            t_child = ts.nodes_time[edge.child]
            t_parent = ts.nodes_time[edge.parent]

            parent_ptr = node_map[edge.parent]
            child_ptr = node_map[edge.child]
            stop_ptr = update_counts_descending_ptr(C, parent_ptr, child_ptr, -1)
            assert np.all(C >= 0)

            num_children_array[edge.parent] -= 1
            last_parent_array[edge.child] = edge.parent

        for edge in edges_in:
            # new edges coming in are those that start at position x
            t_child = ts.nodes_time[edge.child]
            t_parent = ts.nodes_time[edge.parent]

            parent_ptr = node_map[edge.parent]
            child_ptr = node_map[edge.child]
            rec_event = False
            left_parent_time = math.inf
            last_parent = last_parent_array[edge.child]
            if last_parent != -1:
                left_parent_time = ts.nodes_time[last_parent]
                if edge.parent != last_parent:
                    rec_event = True

            min_parent_time = min(left_parent_time, t_parent)
            ret += log_depth_descending(
                C,
                I,
                min_parent_time,
                t_child,
                parent_ptr,
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
            # use current edge to update counts for all subsequent edges
            stop_ptr = update_counts_descending_ptr(C, parent_ptr, child_ptr, 1)
            num_children_array[edge.parent] += 1
            if num_children_array[edge.parent] >= 2:
                coalescent_nodes_array[edge.parent] = 1

    num_coal_events = np.sum(coalescent_nodes_array)
    ret += num_coal_events * np.log(coal_rate)
    return ret


@numba.njit
def _log_likelihood_descending_numba(
    tree_pos,
    I,
    node_map,
    edges_parent,
    edges_child,
    edges_left,
    edges_right,
    rec_rate,
    coal_rate,
):
    ret = 0
    num_nodes = len(node_map)
    coalescent_nodes_array = np.zeros(num_nodes, dtype=np.int8)
    num_children_array = np.zeros(num_nodes, dtype=np.int64)
    C = np.zeros_like(I, dtype=np.int64)

    while tree_pos.next():
        last_parent_array = -np.ones(num_nodes, dtype=np.int64)
        for j in range(tree_pos.out_range[0], tree_pos.out_range[1]):
            e = tree_pos.edge_removal_order[j]
            p = edges_parent[e]
            c = edges_child[e]

            parent_ptr = node_map[p]
            child_ptr = node_map[c]
            stop_ptr = update_counts_descending_ptr(C, parent_ptr, child_ptr, -1)
            num_children_array[p] -= 1
            last_parent_array[c] = p

        for j in range(tree_pos.in_range[0], tree_pos.in_range[1]):
            e = tree_pos.edge_insertion_order[j]
            p = edges_parent[e]
            c = edges_child[e]

            parent_ptr = node_map[p]
            child_ptr = node_map[c]
            t_parent = I[parent_ptr]
            t_child = I[child_ptr]
            rec_event = False
            left_parent_time = np.inf
            last_parent = last_parent_array[c]
            if last_parent != -1:
                last_parent_ptr = node_map[last_parent]
                left_parent_time = I[last_parent_ptr]
                if p != last_parent:
                    rec_event = True

            min_parent_time = min(left_parent_time, t_parent)
            ret += log_depth_descending(
                C,
                I,
                min_parent_time,
                t_child,
                parent_ptr,
                rec_rate,
                coal_rate,
                rec_event,
            )
            ret += log_span(
                rec_rate,
                t_parent,
                t_child,
                edges_left[e],
                edges_right[e],
            )
            # use current edge to update counts for all subsequent edges
            stop_ptr = update_counts_descending_ptr(C, parent_ptr, child_ptr, 1)
            num_children_array[p] += 1
            if num_children_array[p] >= 2:
                coalescent_nodes_array[p] = 1

    num_coal_events = np.sum(coalescent_nodes_array)
    ret += num_coal_events * np.log(coal_rate)

    return ret


def log_likelihood_descending_numba(ts, rec_rate, population_size):
    # here we can no longer account for the fact that past the
    # first mrca we might observe discontinuous edges (for the
    # same parent child pair)
    coal_rate = 1 / (2 * population_size)
    I, node_map = np.unique(ts.nodes_time, return_inverse=True)
    tree_pos = alloc_tree_position(ts)

    return _log_likelihood_descending_numba(
        tree_pos,
        I,
        node_map,
        ts.edges_parent,
        ts.edges_child,
        ts.edges_left,
        ts.edges_right,
        rec_rate,
        coal_rate,
    )
