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


@numba.njit
def update_counts_descending_ptr(counts, start_ptr, stop_ptr, increment):
    assert start_ptr > 0

    i = start_ptr - 1
    while i >= stop_ptr:
        counts[i] += increment
        i -= 1
        if i < 0:
            break
    return i


@numba.njit
def log_depth(
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
        fdiff = f1 - f0
        if fdiff == 0:
            return 0
        n1 = coal_rate / (rec_rate - coal_rate * f1)
        n2 = coal_rate / (rec_rate - coal_rate * f0)
        n3 = fdiff * (rec_rate / coal_rate)
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
        return -coal_rate * cum_area
    
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


@numba.njit
def log_span(r, parent_time, child_time, left, right):
    ret = 0
    if r > 0:
        assert right > left
        assert parent_time > child_time
        ret = -r * (parent_time - child_time) * (right - left)

    return ret


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
    rec_correction
):
    ret = 0
    num_nodes = len(node_map)
    coalescent_nodes_array = np.zeros(num_nodes, dtype=np.int8)
    num_children_array = np.zeros(num_nodes, dtype=np.int64)
    C = np.zeros_like(I, dtype=np.int64)
    last_parent_array = -np.ones(num_nodes, dtype=np.int64)
    visited_array = np.zeros(num_nodes, dtype=np.int8)

    while tree_pos.next():
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
                    if visited_array[p] == 0:
                        rec_event = True
                        if rec_correction:
                            visited_array[p] = 1

            min_parent_time = min(left_parent_time, t_parent)
            ret += log_depth(
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


def log_likelihood(ts, rec_rate, population_size, rec_correction=False):
    # here we can no longer account for the fact that past the
    # first mrca we might observe discontinuous edges (for the
    # same parent child pair)
    coal_rate = 1 / (2 * population_size)
    I, node_map = np.unique(ts.nodes_time, return_inverse=True)
    tree_pos = alloc_tree_position(ts)

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
        rec_correction
    )
