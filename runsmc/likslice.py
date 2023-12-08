import itertools
import math
import numpy as np
import numba
import tskit

import runsmc.liknb as liknb
import runsmc.likstepwise as likstep


@numba.njit
def log_depth(
    left_count,
    intervals,
    min_parent_time,
    child_ptr,
    parent_ptr,
    rec_rate,
    coal_rate,
    rec_event,
    min_time_ptr,
    max_time_ptr
):
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
        if i < max_time_ptr:
            b = np.exp(-rec_rate * intervals[i] - cum_area)
            temp = a * b
            (s, e) = likstep.two_sum(s, temp)
            t += e
            
            temp = -a_prev * b
            (s, e) = likstep.two_sum(s, temp)
            t += e
            a = a_prev

        # adapt cum_area by sliding 1 interval to left
        cum_area += (
            (intervals[i] - intervals[i - 1]) * left_count[i - 1] * coal_rate[i - 1]
        )
        i -= 1
        if i == child_ptr:
            temp = a * np.exp(-rec_rate * intervals[i] - cum_area)
            (s, e) = likstep.two_sum(s, temp)
            t += e
            break
        if i == min_time_ptr:
            break
        a_prev = rec_rate / (rec_rate - coal_rate[i - 1] * left_count[i - 1])

    total = s + t
    if total > 0:
        return np.log(total)
    elif total==0:
        return 0
    else:
        assert False


@numba.njit
def _log_likelihood(
    tree_pos,
    I,
    node_map,
    edges_parent,
    edges_child,
    edges_left,
    edges_right,
    rec_rate,
    coal_rate,
    min_time_ptr,
    max_time_ptr,
    alpha
):
    ret = 0
    num_nodes = len(node_map)
    coalescent_interval_array = np.zeros(I.size, dtype=np.int8)
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
            if (child_ptr < max_time_ptr and parent_ptr > min_time_ptr):
                ret += log_depth(
                    C,
                    I,
                    min_parent_time,
                    child_ptr,
                    parent_ptr,
                    rec_rate,
                    coal_rate,
                    rec_event,
                    min_time_ptr,
                    max_time_ptr,
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
            if num_children_array[p] == 2:
                coalescent_interval_array[parent_ptr] += 1

    for i in range(coalescent_interval_array.size):
        coalescent_interval_array[:min_time_ptr] = 0
        coalescent_interval_array[max_time_ptr + alpha:] = 0
        ret += coalescent_interval_array[i] * np.log(coal_rate[i])

    return ret


def log_likelihood(ts, rec_rate, population_size, time_slice, ploidy=2):
    # here we can no longer account for the fact that past the
    # first mrca we might observe discontinuous edges (for the
    # same parent child pair)
    # add time_slice to ts.nodes_time
    
    all_nodes_time = np.hstack([ts.nodes_time, np.array(time_slice)])
    I, node_map = np.unique(all_nodes_time, return_inverse=True)
    coal_rate = np.full(I.size, 1 / (ploidy * population_size))
    tree_pos = liknb.alloc_tree_position(ts)
    min_time_ptr = node_map[-2]
    max_time_ptr = node_map[-1]
    alpha = 1 # this should check whether time_slice[1]
    # coincides with an existing node.
    return _log_likelihood(
        tree_pos,
        I,
        node_map,
        ts.edges_parent,
        ts.edges_child,
        ts.edges_left,
        ts.edges_right,
        rec_rate,
        coal_rate,
        min_time_ptr,
        max_time_ptr,
        alpha
    )