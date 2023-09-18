import itertools
import math
import numpy as np
import numba
import tskit

import runsmc.liknb as liknb
import runsmc.fenwick as fenwick


@numba.njit(cache=True)
def log_depth_descending_comp(
    left_count,
    intervals,
    min_parent_time,
    child_time,
    parent_ptr,
    rec_rate,
    coal_rate,
    rec_event,
):
    cum_area = 0
    i = parent_ptr
    stop_time = child_time
    
    while intervals[i] > stop_time:
        span = intervals[i] - intervals[i - 1]
        cum_area += span * left_count[i - 1]
        i -= 1
        if i < 1:
            break

    return cum_area


@numba.njit(cache=True)
def _log_likelihood_descending_numba_comp(
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
    last_parent_array = -np.ones(num_nodes, dtype=np.int64)
    F = np.zeros_like(I, dtype=np.float64)

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
            ret += log_depth_descending_comp(
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
                edges_left[e],
                edges_right[e],
            )
            # use current edge to update counts for all subsequent edges
            stop_ptr = liknb.update_counts_descending_ptr(C, parent_ptr, child_ptr, 1)
            num_children_array[p] += 1
            if num_children_array[p] >= 2:
                coalescent_nodes_array[p] = 1

        for j in range(tree_pos.out_range[0], tree_pos.out_range[1]):
            e = tree_pos.edge_removal_order[j]
            c = edges_child[e]
            last_parent_array[c] = -1

    num_coal_events = np.sum(coalescent_nodes_array)
    ret += num_coal_events * np.log(coal_rate)

    return ret


def log_likelihood_descending_numba(ts, rec_rate, population_size):
    # here we can no longer account for the fact that past the
    # first mrca we might observe discontinuous edges (for the
    # same parent child pair)
    coal_rate = 1 / (2 * population_size)
    I, node_map = np.unique(ts.nodes_time, return_inverse=True)
    tree_pos = liknb.alloc_tree_position(ts)

    return _log_likelihood_descending_numba_comp(
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


@numba.njit(cache=True)
def update_fenwick_descending_ptr(F, n, intervals, parent_ptr, child_ptr, v):
    ptr = parent_ptr
    while ptr > child_ptr:
        fenwick.updatebit(F, n, ptr, v*intervals[ptr])
        ptr -= 1

@numba.njit(cache=True)
def log_depth_bit(
    bi_tree,
    child_ptr,
    parent_ptr
):
    return fenwick.get_range_sum(bi_tree, child_ptr, parent_ptr)

@numba.njit(cache=True)
def _log_likelihood_descending_numba_bit(
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
    
    intervals = I[1:] - I[:-1]
    ret = 0
    num_nodes = len(node_map)
    coalescent_nodes_array = np.zeros(num_nodes, dtype=np.int8)
    num_children_array = np.zeros(num_nodes, dtype=np.int64)
    C = np.zeros_like(I, dtype=np.int64)
    last_parent_array = -np.ones(num_nodes, dtype=np.int64)
    F = np.zeros(C.size + 1, dtype=np.float64)
    n = F.size

    while tree_pos.next():
        for j in range(tree_pos.out_range[0], tree_pos.out_range[1]):
            e = tree_pos.edge_removal_order[j]
            p = edges_parent[e]
            c = edges_child[e]

            parent_ptr = node_map[p]
            child_ptr = node_map[c]
            update_fenwick_descending_ptr(F, n, intervals, parent_ptr, child_ptr, -1)
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
            ret += log_depth_bit(
                F,
                child_ptr,
                parent_ptr-1
                )
            ret += liknb.log_span(
                rec_rate,
                t_parent,
                t_child,
                edges_left[e],
                edges_right[e],
            )
            update_fenwick_descending_ptr(F, n, intervals, parent_ptr, child_ptr, 1)

            num_children_array[p] += 1
            if num_children_array[p] >= 2:
                coalescent_nodes_array[p] = 1

        for j in range(tree_pos.out_range[0], tree_pos.out_range[1]):
            e = tree_pos.edge_removal_order[j]
            c = edges_child[e]
            last_parent_array[c] = -1

    num_coal_events = np.sum(coalescent_nodes_array)
    ret += num_coal_events * np.log(coal_rate)

    return ret


def log_likelihood_descending_numba_bit(ts, rec_rate, population_size):
    # here we can no longer account for the fact that past the
    # first mrca we might observe discontinuous edges (for the
    # same parent child pair)
    coal_rate = 1 / (2 * population_size)
    I, node_map = np.unique(ts.nodes_time, return_inverse=True)
    tree_pos = liknb.alloc_tree_position(ts)

    return _log_likelihood_descending_numba_bit(
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


