import itertools
import msprime
import math
import numpy as np
import operator
import tskit


def update_array(array, intervals, t_parent, t_child):
    i = 0
    while i < array.size:
        if intervals[i] >= t_parent:
            break
        if intervals[i] >= t_child:
            array[i] += 1
        i += 1


def count_lineages(
    left,
    roots,
    focal_child,
    edge_array,
    left_child_array,
    right_sib_array,
    edges_left,
    times,
    intervals,
):
    f = np.zeros(len(intervals) - 1, dtype=np.int64)
    stack = roots
    while stack:
        parent = stack.pop()
        child = left_child_array[parent]
        while child != tskit.NULL:
            tc = times[child]
            tp = times[parent]
            edge_id = edge_array[child]
            if edges_left[edge_id] < left or child < focal_child:
                update_array(f, intervals, tp, tc)
            if tc > intervals[0]:
                stack.append(child)
            child = right_sib_array[child]

    return f


def lineages_to_left_count(edge, ts):
    # returns values ordered from tc to tp
    tp = ts.nodes_time[edge.parent]
    tc = ts.nodes_time[edge.child]
    dts = ts.decapitate(time=tp)
    tree = dts.at(edge.left)
    intervals = [tp]
    for node in tree.nodes(order="timedesc"):
        node_time = tree.time(node)
        if node_time >= tc and node_time < tp:
            intervals.append(node_time)
            if node_time == 0:
                break
    intervals = np.array(intervals)[::-1]
    f = count_lineages(
        edge.left,
        tree.roots,
        edge.child,
        tree.edge_array,
        tree.left_child_array,
        tree.right_sib_array,
        dts.edges_left,
        dts.nodes_time,
        intervals,
    )

    return f, intervals


def coalescencing_child(tree, ts, parent):
    """
    Returns the index of the child associated with the
    edge with the largest left coordinate or with the largest
    node index to break ties.
    Lineages can only coalesce with lineages with that start off
    to their left. Or have a smaller node id to break ties.
    """
    left = 0
    coal_child = -1
    for child in tree.children(parent):
        edge = tree.edge(child)
        edge_left = ts.edges_left[edge]
        if edge_left >= left:
            left = edge_left
            if edge_left == left:
                coal_child = max(coal_child, child)
            else:
                coal_child = child

    assert coal_child < math.inf
    return coal_child


def edges_by_child_timeasc(tables):
    # sort edges by child age, then edge.left to break ties.
    idxs = tables.nodes.time[tables.edges.child]
    idxs2 = tables.edges.left
    it = (tables.edges[u] for u in np.lexsort((idxs2, tables.edges.child, idxs)))
    return itertools.groupby(it, operator.attrgetter("child"))


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
        n = rec_rate * coal_rate * (f1 - f0)
        d = (rec_rate - coal_rate * f1) * (rec_rate - coal_rate * f0)
        return n / d

    if not rec_event:
        # if no recombination event expression simplifies to
        ret = np.exp(-coal_rate * cum_area)
    else:
        denoms = rec_rate - coal_rate * left_count
        if np.any(denoms == 0):
            raise ValueError("denom is 0")
        else:
            t1 = intervals[0]
            ret = rec_rate / denoms[0] * np.exp(-rec_rate * t1 - coal_rate * cum_area)
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
                rec_rate / denoms[i - 1] * np.exp(-rec_rate * t1 - coal_rate * cum_area)
            )
    assert ret > 0, "About to take log of non-positive value."
    return np.log(ret)


def log_span(r, parent_time, child_time, left, right):
    ret = 0
    if r > 0:
        ret = -r * (parent_time - child_time) * (right - left)

    return ret


def log_edge(
    left,
    right,
    min_parent_time,
    f,
    intervals,
    rec_rate,
    coal_rate,
    rec_event,
    coal_event,
):
    """
    Compute the likelihood for a certain edge:
    This has 3 components:
    1/ log_depth: likelihood of the left endpoint of the segment,
    spanned by that edge, not coalescing with any of the lineages
    it can coalesce with until the coalescence event in edge.parent.
    Here we integrate out the time of the recombination event
    (in case of a recombination) that happened at position edge.left
    2/ if coal_event: if the coalescence happend with another segment
    that was among the segments that edge.child could coalesce with
    we add this additional term.
    3/ log_span: likelihood of observing an edge of this length
    """
    ret = 0
    ret += log_depth(
        min_parent_time,
        f,
        intervals,
        rec_rate,
        coal_rate,
        rec_event,
    )
    if coal_event:
        ret += np.log(coal_rate) #* f[-1]
    # -r (t_p - t_c) * (r - l)
    right_parent_time = intervals[-1]
    child_time = intervals[0]
    ret += log_span(rec_rate, right_parent_time, child_time, right, left)

    return ret


def log_likelihood(tables, rec_rate, population_size):
    # assumption: tables have been generated with
    # coalescing_segments_only flag set to False

    ret = 0
    ts = tables.tree_sequence()
    tree = ts.first()
    num_nodes = tables.nodes.num_rows
    coal_rate = 1/(2 * population_size)

    # sort edges based on child, edge.left to break ties
    for child, edges in edges_by_child_timeasc(tables):
        left_parent_time = math.inf
        rec_event = False
        last_parent = -1
        child_time = tables.nodes[child].time

        for edge in edges:
            if edge.parent != last_parent:
                if last_parent == -1:
                    left = edge.left
                    right = edge.right
                else:
                    # all information for previous parent has been collected
                    # perform computations for previous
                    ret += log_edge(
                        left,
                        right,
                        min_parent_time,
                        f,
                        intervals,
                        rec_rate,
                        coal_rate,
                        rec_event,
                        coal_event,
                    )

                    rec_event = True
                    left_parent_time = right_parent_time

                right_parent_time = tables.nodes[edge.parent].time
                f, intervals = lineages_to_left_count(
                    edge,
                    ts,
                )
                last_parent = edge.parent
                tree.seek(edge.left)
                if f[-1] > 0:
                    coal_event = child == coalescencing_child(tree, ts, last_parent)
                else:
                    coal_event = False
                min_parent_time = min(left_parent_time, right_parent_time)
            else:
                right = edge.right

        ret += log_edge(
            left,
            right,
            min_parent_time,
            f,
            intervals,
            rec_rate,
            coal_rate,
            rec_event,
            coal_event,
        )

    return ret
