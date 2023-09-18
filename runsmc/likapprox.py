import itertools
import math
import numpy as np
import numba
import tskit

import runsmc.liknb as liknb


def log_depth_descending_approx(
    C, cum_area, min_parent_t, child_t, parent_t, rec_rate, coal_rate, rec_event, order=1
):
    if not rec_event:
        return -coal_rate * cum_area   
    if order == 1:
        avg_rate = cum_area / (parent_t - child_t)
        if avg_rate == rec_rate / coal_rate:
            return (
                np.log(coal_rate * rec_rate * (-child_t + min_parent_t))
                - rec_rate * parent_t
            )
        else:
            num1 = coal_rate * rec_rate
            num2a = np.exp(
                avg_rate * coal_rate * child_t
                - avg_rate * coal_rate * parent_t
                - rec_rate * child_t
            )
            num2b = -np.exp(
                avg_rate * coal_rate * min_parent_t
                - avg_rate * coal_rate * parent_t
                - rec_rate * min_parent_t
            )
            denom = -avg_rate * coal_rate + rec_rate
            return np.log(num1 * (num2a + num2b) / denom)
    else:
        raise NotImplementedError


def _log_likelihood_descending_numba_approx(
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
    """
    Function should perform all counts and return the parameters for an
    approximation of the rate function for each edge.
    """
    ret = 0
    num_nodes = len(node_map)
    coalescent_nodes_array = np.zeros(num_nodes, dtype=np.int8)
    num_children_array = np.zeros(num_nodes, dtype=np.int64)
    C = np.zeros_like(I, dtype=np.int64)
    last_parent_array = -np.ones(num_nodes, dtype=np.int64)
    intervals = I[1:] - I[:-1]

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
            cum_area = np.sum(
                intervals[child_ptr:parent_ptr] * C[child_ptr:parent_ptr]
            )
            ret += log_depth_descending_approx(
                C,
                cum_area,
                min_parent_time,
                t_child,
                t_parent,
                rec_rate,
                coal_rate,
                rec_event,
                order=1
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


def log_likelihood_descending_numba_approx(ts, rec_rate, population_size):
    # here we can no longer account for the fact that past the
    # first mrca we might observe discontinuous edges (for the
    # same parent child pair)
    coal_rate = 1 / (2 * population_size)
    I, node_map = np.unique(ts.nodes_time, return_inverse=True)
    tree_pos = liknb.alloc_tree_position(ts)

    return _log_likelihood_descending_numba_approx(
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
