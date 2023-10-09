import itertools
import math
import numpy as np
import numba
import tskit

from scipy.special import logsumexp
from runsmc import liknb

@numba.njit(cache=True)
def two_sum(a, b):
    x = a + b
    y = x - a
    e = a - (x - y) + (b - y)
    return x, e


@numba.njit(cache=True)
def log_depth_descending(
    left_count,
    intervals,
    min_parent_time,
    child_ptr,
    parent_ptr,
    rec_rate,
    coal_rate,
    rec_event,
):
    ret = np.zeros(parent_ptr - child_ptr, dtype=np.float64)
    cum_area = 0
    i = parent_ptr

    if not rec_event:
        stop_time = intervals[child_ptr]
    else:
        stop_time = min_parent_time

    # update cumulative area remaining under left_count * coal_rate
    # while moving from t_parent to t_child
    # if recombination: stop at min_parent time
    # else: simplify compute cumulative area
    while intervals[i] > stop_time:
        span = intervals[i] - intervals[i - 1]
        cum_area += span * left_count[i - 1] * coal_rate[i - 1]
        i -= 1
        if i < 1:
            break

    if not rec_event:
        return -cum_area

    assert min_parent_time == intervals[i]
    s, t = 0, 0
    a_prev = rec_rate / (rec_rate - coal_rate[i - 1] * left_count[i - 1])
    a = 0
    while i > 0:
        b = np.exp(-rec_rate * intervals[i] - cum_area)
        # ret += a * b - a_prev * b
        temp = a * b
        (s, e) = two_sum(s, temp)
        t += e
        
        temp = -a_prev * b
        (s, e) = two_sum(s, temp)
        t += e
        
        # adapt cum_area by sliding 1 interval to left
        cum_area += (
            (intervals[i] - intervals[i - 1]) * left_count[i - 1] * coal_rate[i - 1]
        )
        i -= 1
        a = a_prev
        if i == child_ptr:
            # ret += a * b
            temp = a * np.exp(-rec_rate * intervals[i] - cum_area)
            (s, e) = two_sum(s, temp)
            t += e
            break
        a_prev = rec_rate / (rec_rate - coal_rate[i - 1] * left_count[i - 1])

    total = s + t
    assert total > 0
    return np.log(total)


@numba.njit(cache=True)
def _log_likelihood_stepwise_ne(
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
    coalescent_interval_array = np.zeros(coal_rate.size, dtype=np.int8)
    num_children_array = np.zeros(num_nodes, dtype=np.int64)
    C = np.zeros_like(I, dtype=np.int64)
    last_parent_array = -np.ones(num_nodes, dtype=np.int64)

    while tree_pos.next():
        for j in range(tree_pos.out_range[0], tree_pos.out_range[1]):
            e = tree_pos.edge_removal_order[j]
            p = edges_parent[e]
            c = edges_child[e]

            parent_ptr = node_map[p]
            child_ptr = node_map[c]
            stop_ptr = liknb.update_counts_descending_ptr(C, parent_ptr, child_ptr, -1)
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
                child_ptr,
                parent_ptr,
                rec_rate,
                coal_rate,
                rec_event,
            )
            ret += liknb.log_span(
                rec_rate,
                t_parent,
                t_child,
                edges_left[e],
                edges_right[e],
            )
            # use current edge to update counts for all subsequent edges
            stop_ptr = liknb.update_counts_descending_ptr(C, parent_ptr, child_ptr, 1)
            num_children_array[p] += 1
            if num_children_array[p] >= 2:
                coalescent_interval_array[parent_ptr] += 1

        for j in range(tree_pos.out_range[0], tree_pos.out_range[1]):
            e = tree_pos.edge_removal_order[j]
            c = edges_child[e]
            last_parent_array[c] = -1

    for i in range(coalescent_interval_array.size):
        ret += coalescent_interval_array[i] * np.log(coal_rate[i])

    return ret


@numba.njit(cache=True)
def update_mapping(j, k, old_map, new_map):
    current_value = old_map[k]
    while current_value == old_map[k]:
        new_map[k] = old_map[k] + j
        k += 1
        if k == old_map.size:
            break
    return k


@numba.njit(cache=True)
def merge_time_arrays(node_times, step_times, node_mapping):
    """
    Mapping should be pointing to
    Returns merged version of a and b, sorted, with only unique
    elements.
    Update mapping adapt the
    """
    a = np.sort(node_times)
    b = np.sort(step_times)
    merged = np.zeros(a.size + b.size)
    new_node_mapping = np.zeros(node_mapping.size, dtype=np.int64)
    step_mapping = np.zeros(merged.size, dtype=np.int64)
    assert step_times[0] <= node_times[0], "There is a gap in the step times."

    i, j, k, d = 0, 0, 0, 0
    while i < a.size and j < b.size:
        if a[i] < b[j]:
            merged[d] = a[i]
            k = update_mapping(d - i, k, node_mapping, new_node_mapping)
            i += 1
        elif a[i] > b[j]:
            merged[d] = b[j]
            j += 1
        else:
            merged[d] = a[i]
            k = update_mapping(d - i, k, node_mapping, new_node_mapping)
            i += 1
            j += 1
        step_mapping[d] = j - 1
        d += 1

    while i < a.size:
        merged[d] = a[i]
        step_mapping[d] = j - 1
        k = update_mapping(d - i, k, node_mapping, new_node_mapping)
        i += 1
        d += 1

    while j < b.size:
        merged[d] = b[j]
        j += 1
        step_mapping[d] = j - 1
        d += 1
    if d < i + j:
        merged = merged[: d - (i + j)]
        step_mapping = step_mapping[: d - (i + j)]

    return merged, new_node_mapping, step_mapping


def log_likelihood_stepwise_ne(ts, rec_rate, ne_steps, time_steps, ploidy=2):
    # here we can no longer account for the fact that past the
    # first mrca we might observe discontinuous edges (for the
    # same parent child pair)
    assert ne_steps.size == time_steps.size
    assert time_steps[0] == 0.0
    coal_rate_steps = 1 / (ploidy * ne_steps)
    I, node_map = np.unique(ts.nodes_time, return_inverse=True)
    I, node_map, rate_map = merge_time_arrays(I, time_steps, node_map)
    coal_rate = coal_rate_steps[rate_map]
    tree_pos = liknb.alloc_tree_position(ts)

    return _log_likelihood_stepwise_ne(
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
