import pytest
import tskit
import msprime
import numpy as np

import runsmc.likelihoods as lik


class TestCountLineages:
    def test_lineages_to_left(self):
        # 2.36┊                            39                     ┊
        #     ┊                 ┏━━━━━━━━━━━┻━━━━━━━━━━┓          ┊
        # 1.86┊                38                      ┃          ┊
        #     ┊        ┏━━━━━━━━┻━━━━━━━┓              ┃          ┊
        # 1.12┊        ┃                ┃             37          ┊
        #     ┊        ┃                ┃         ┏━━━━┻━━━━┓     ┊
        # 1.02┊        ┃                ┃         ┃         ┃     ┊
        #     ┊        ┃                ┃         ┃         ┃     ┊
        # 0.83┊       35                ┃         ┃         ┃     ┊
        #     ┊ ┏━━━━━━┻━━━━━━┓         ┃         ┃         ┃     ┊
        # 0.47┊ ┃             ┃        34         ┃         ┃     ┊
        #     ┊ ┃             ┃    ┏━━━━┻━━━┓     ┃         ┃     ┊
        # 0.33┊ ┃            33    ┃        ┃     ┃         ┃     ┊
        #     ┊ ┃         ┏━━━┻━━┓ ┃        ┃     ┃         ┃     ┊
        # 0.29┊ ┃        32      ┃ ┃        ┃     ┃         ┃     ┊
        #     ┊ ┃      ┏━━┻━━━┓  ┃ ┃        ┃     ┃         ┃     ┊
        # 0.25┊ ┃      ┃      ┃  ┃ ┃       31     ┃         ┃     ┊
        #     ┊ ┃      ┃      ┃  ┃ ┃     ┏━━┻━━┓  ┃         ┃     ┊
        # 0.17┊ ┃      ┃      ┃  ┃ ┃     ┃     ┃ 30         ┃     ┊
        #     ┊ ┃      ┃      ┃  ┃ ┃     ┃     ┃ ┏┻━┓       ┃     ┊
        # 0.16┊ ┃      ┃      ┃  ┃ ┃     ┃     ┃ ┃  ┃      29     ┊
        #     ┊ ┃      ┃      ┃  ┃ ┃     ┃     ┃ ┃  ┃    ┏━━┻━┓   ┊
        # 0.11┊ ┃      ┃      ┃  ┃ ┃     ┃     ┃ ┃  ┃    ┃   28   ┊
        #     ┊ ┃      ┃      ┃  ┃ ┃     ┃     ┃ ┃  ┃    ┃   ┏┻━┓ ┊
        # 0.09┊ ┃     27      ┃  ┃ ┃     ┃     ┃ ┃  ┃    ┃   ┃  ┃ ┊
        #     ┊ ┃   ┏━━┻━━┓   ┃  ┃ ┃     ┃     ┃ ┃  ┃    ┃   ┃  ┃ ┊
        # 0.07┊ ┃   ┃     ┃   ┃  ┃ ┃    26     ┃ ┃  ┃    ┃   ┃  ┃ ┊
        #     ┊ ┃   ┃     ┃   ┃  ┃ ┃  ┏━━┻━━┓  ┃ ┃  ┃    ┃   ┃  ┃ ┊
        # 0.06┊ ┃   ┃     ┃   ┃  ┃ ┃  ┃     ┃  ┃ ┃ 25    ┃   ┃  ┃ ┊
        #     ┊ ┃   ┃     ┃   ┃  ┃ ┃  ┃     ┃  ┃ ┃ ┏┻━┓  ┃   ┃  ┃ ┊
        # 0.05┊ ┃   ┃    24   ┃  ┃ ┃  ┃     ┃  ┃ ┃ ┃  ┃  ┃   ┃  ┃ ┊
        #     ┊ ┃   ┃   ┏━┻┓  ┃  ┃ ┃  ┃     ┃  ┃ ┃ ┃  ┃  ┃   ┃  ┃ ┊
        # 0.04┊ ┃   ┃   ┃  ┃  ┃  ┃ ┃ 23     ┃  ┃ ┃ ┃  ┃  ┃   ┃  ┃ ┊
        #     ┊ ┃   ┃   ┃  ┃  ┃  ┃ ┃ ┏┻┓    ┃  ┃ ┃ ┃  ┃  ┃   ┃  ┃ ┊
        # 0.04┊ ┃  22   ┃  ┃  ┃  ┃ ┃ ┃ ┃    ┃  ┃ ┃ ┃  ┃  ┃   ┃  ┃ ┊
        #     ┊ ┃ ┏━┻┓  ┃  ┃  ┃  ┃ ┃ ┃ ┃    ┃  ┃ ┃ ┃  ┃  ┃   ┃  ┃ ┊
        # 0.03┊ ┃ ┃  ┃  ┃  ┃  ┃  ┃ ┃ ┃ ┃    ┃  ┃ ┃ ┃  ┃ 21   ┃  ┃ ┊
        #     ┊ ┃ ┃  ┃  ┃  ┃  ┃  ┃ ┃ ┃ ┃    ┃  ┃ ┃ ┃  ┃ ┏┻┓  ┃  ┃ ┊
        # 0.01┊ ┃ ┃  ┃  ┃  ┃  ┃  ┃ ┃ ┃ ┃   20  ┃ ┃ ┃  ┃ ┃ ┃  ┃  ┃ ┊
        #     ┊ ┃ ┃  ┃  ┃  ┃  ┃  ┃ ┃ ┃ ┃  ┏━┻┓ ┃ ┃ ┃  ┃ ┃ ┃  ┃  ┃ ┊
        # 0.00┊ 0 3 12 11 13 15 18 2 6 9 10 14 7 1 4 16 5 8 17 19 ┊
        #  0                                                   1
        #                                     39            ┊
        #                       ┏━━━━━━━━━━┻━━━━━━━━━━┓     ┊
        #                      38                     ┃     ┊
        #               ┏━━━━━━━┻━━━━━━━┓             ┃     ┊
        #               ┃               ┃             ┃     ┊
        #               ┃               ┃             ┃     ┊
        #              36               ┃             ┃     ┊
        #        ┏━━━━━━┻━━━━━┓         ┃             ┃     ┊
        #       35            ┃         ┃             ┃     ┊
        # ┏━━━━━━┻━━━━━━┓     ┃         ┃             ┃     ┊
        # ┃             ┃     ┃        34             ┃     ┊
        # ┃             ┃     ┃     ┏━━━┻━━━━┓        ┃     ┊
        # ┃            33     ┃     ┃        ┃        ┃     ┊
        # ┃         ┏━━━┻━━┓  ┃     ┃        ┃        ┃     ┊
        # ┃        32      ┃  ┃     ┃        ┃        ┃     ┊
        # ┃      ┏━━┻━━━┓  ┃  ┃     ┃        ┃        ┃     ┊
        # ┃      ┃      ┃  ┃  ┃     ┃       31        ┃     ┊
        # ┃      ┃      ┃  ┃  ┃     ┃     ┏━━┻━━┓     ┃     ┊
        # ┃      ┃      ┃  ┃ 30     ┃     ┃     ┃     ┃     ┊
        # ┃      ┃      ┃  ┃ ┏┻━┓   ┃     ┃     ┃     ┃     ┊
        # ┃      ┃      ┃  ┃ ┃  ┃   ┃     ┃     ┃    29     ┊
        # ┃      ┃      ┃  ┃ ┃  ┃   ┃     ┃     ┃  ┏━━┻━┓   ┊
        # ┃      ┃      ┃  ┃ ┃  ┃   ┃     ┃     ┃  ┃   28   ┊
        # ┃      ┃      ┃  ┃ ┃  ┃   ┃     ┃     ┃  ┃   ┏┻━┓ ┊
        # ┃     27      ┃  ┃ ┃  ┃   ┃     ┃     ┃  ┃   ┃  ┃ ┊
        # ┃   ┏━━┻━━┓   ┃  ┃ ┃  ┃   ┃     ┃     ┃  ┃   ┃  ┃ ┊
        # ┃   ┃     ┃   ┃  ┃ ┃  ┃   ┃    26     ┃  ┃   ┃  ┃ ┊
        # ┃   ┃     ┃   ┃  ┃ ┃  ┃   ┃  ┏━━┻━┓   ┃  ┃   ┃  ┃ ┊
        # ┃   ┃     ┃   ┃  ┃ ┃ 25   ┃  ┃    ┃   ┃  ┃   ┃  ┃ ┊
        # ┃   ┃     ┃   ┃  ┃ ┃ ┏┻━┓ ┃  ┃    ┃   ┃  ┃   ┃  ┃ ┊
        # ┃   ┃    24   ┃  ┃ ┃ ┃  ┃ ┃  ┃    ┃   ┃  ┃   ┃  ┃ ┊
        # ┃   ┃   ┏━┻┓  ┃  ┃ ┃ ┃  ┃ ┃  ┃    ┃   ┃  ┃   ┃  ┃ ┊
        # ┃   ┃   ┃  ┃  ┃  ┃ ┃ ┃  ┃ ┃ 23    ┃   ┃  ┃   ┃  ┃ ┊
        # ┃   ┃   ┃  ┃  ┃  ┃ ┃ ┃  ┃ ┃ ┏┻┓   ┃   ┃  ┃   ┃  ┃ ┊
        # ┃  22   ┃  ┃  ┃  ┃ ┃ ┃  ┃ ┃ ┃ ┃   ┃   ┃  ┃   ┃  ┃ ┊
        # ┃ ┏━┻┓  ┃  ┃  ┃  ┃ ┃ ┃  ┃ ┃ ┃ ┃   ┃   ┃  ┃   ┃  ┃ ┊
        # ┃ ┃  ┃  ┃  ┃  ┃  ┃ ┃ ┃  ┃ ┃ ┃ ┃   ┃   ┃ 21   ┃  ┃ ┊
        # ┃ ┃  ┃  ┃  ┃  ┃  ┃ ┃ ┃  ┃ ┃ ┃ ┃   ┃   ┃ ┏┻┓  ┃  ┃ ┊
        # ┃ ┃  ┃  ┃  ┃  ┃  ┃ ┃ ┃  ┃ ┃ ┃ ┃  20   ┃ ┃ ┃  ┃  ┃ ┊
        # ┃ ┃  ┃  ┃  ┃  ┃  ┃ ┃ ┃  ┃ ┃ ┃ ┃  ┏┻━┓ ┃ ┃ ┃  ┃  ┃ ┊
        # 0 3 12 11 13 15 18 1 4 16 2 6 9 10 14 7 5 8 17 19 ┊

        ts = msprime.sim_ancestry(
            samples=10,
            recombination_rate=0.05,
            population_size=1,
            sequence_length=10,
            model="smc",
            coalescing_segments_only=False,
            random_seed=123,
        )
        assert ts.num_trees == 5
        tables = ts.dump_tables()
        tables.keep_intervals(np.array([[0, 5]]))
        tables.rtrim()
        ts = tables.tree_sequence()
        edge_id = 32
        edge = tables.edges[edge_id]
        parent = 36
        child = 30
        assert edge.child == child
        assert edge.parent == parent
        assert edge.left == 1
        tp = tables.nodes[edge.parent].time
        tc = tables.nodes[edge.child].time
        intervals = [
            0.17267483,
            0.24757562,
            0.28614193,
            0.33072171,
            0.468715,
            1.02148028,
        ]
        f, g = lik.lineages_to_left_count(edge, tp, tc, ts, intervals)
        f_exp = np.array([7, 6, 5, 4, 3, 1])
        g_exp = np.array([2, 2, 2, 2, 2, 3])
        assert np.array_equal(f, f_exp)

    def no_test_update_sort(self):
        f = np.zeros(4, dtype=np.int64)
        intervals = np.random.rand(4)
        intervals.sort()
        intervals[-1] = 0
        print(intervals)
        t_child = np.random.rand(1)[0]
        print(t_child)
        insert_idx = len(intervals) - 1
        lik.update_and_sort(f, intervals, insert_idx, t_child)
        print(intervals)
        print(f)
        assert False

    def no_test_update_sort_integers(self):
        f = np.array([1, 2, 4, 0])
        intervals = np.array([1, 5, 7, 0])
        t_child = 5
        insert_idx = len(intervals) - 1
        lik.update_and_sort(f, intervals, insert_idx, t_child)
        assert np.array_equal(intervals, np.array([1, 5, 7, 0]))
        print(f)
        assert np.array_equal(f, np.array([2, 3, 4, 0]))

    def no_test_update_sort_integers2(self):
        f = np.array([5, 1, 2, 0])
        intervals = np.array([1, 5, 7, 0])
        t_child = 2
        insert_idx = len(intervals) - 1
        lik.update_and_sort(f, intervals, insert_idx, t_child)
        assert np.array_equal(intervals, np.array([1, 2, 5, 7]))
        print(f)
        assert np.array_equal(f, np.array([6, 7, 1, 2]))

    def no_test_update_sort_integers3(self):
        f = np.array([4, 9, 1, 0])
        intervals = np.array([1, 5, 7, 0])
        t_child = 1
        insert_idx = len(intervals) - 1
        lik.update_and_sort(f, intervals, insert_idx, t_child)
        assert np.array_equal(intervals, np.array([1, 5, 7, 0]))
        assert np.array_equal(f, np.array([5, 9, 1, 0]))


class TestRunSMC:
    def run_smc(r, seed):
        ts = msprime.sim_ancestry(
            samples=10,
            recombination_rate=r,
            population_size=1,
            sequence_length=10,
            model="smc",
            coalescing_segments_only=False,
            random_seed=seed,
        )
        return ts
