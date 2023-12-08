import itertools
import math
import numpy as np
import numba
import tskit

import runsmc.liknb as liknb

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
    last_parent_array = -np.ones(ts.num_nodes, dtype=np.int64)

    for _, edges_out, edges_in in ts.edge_diffs():

        for edge in edges_out:
            t_child = ts.nodes_time[edge.child]
            t_parent = ts.nodes_time[edge.parent]

            parent_ptr = node_map[edge.parent]
            child_ptr = node_map[edge.child]
            stop_ptr = liknb.update_counts_descending_ptr(C, parent_ptr, child_ptr, -1)
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
            ret += liknb.log_depth(
                C,
                I,
                min_parent_time,
                t_child,
                parent_ptr,
                rec_rate,
                coal_rate,
                rec_event,
            )
            ret += liknb.log_span(
                rec_rate,
                t_parent,
                t_child,
                edge.left,
                edge.right,
            )
            # use current edge to update counts for all subsequent edges
            stop_ptr = liknb.update_counts_descending_ptr(C, parent_ptr, child_ptr, 1)
            num_children_array[edge.parent] += 1
            if num_children_array[edge.parent] >= 2:
                coalescent_nodes_array[edge.parent] = 1

    num_coal_events = np.sum(coalescent_nodes_array)
    ret += num_coal_events * np.log(coal_rate)
    return ret