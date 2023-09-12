import msprime
import math
import numpy as np
import pytest
import tskit

import runsmc.likelihoods as lik
import runsmc.liknb as liknb


def run_hudson(r, pop_size, seed, num_samples=10):
    ts = msprime.sim_ancestry(
        samples=num_samples,
        recombination_rate=r,
        population_size=pop_size,
        sequence_length=100,
        coalescing_segments_only=False,
        random_seed=seed,
        ploidy=2,
    )
    return ts


def run_smc(r, pop_size, seed, samples=10):
    ts = msprime.sim_ancestry(
        samples=samples,
        recombination_rate=r,
        population_size=pop_size,
        sequence_length=100,
        model="smc",
        coalescing_segments_only=False,
        random_seed=seed,
        ploidy=2,
    )
    return ts


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
        intervals_exp = np.array([ts.nodes_time[u] for u in range(30, 37)])
        f_exp = np.array([8, 7, 6, 5, 4, 2])
        self.verify_edge(ts, edge_id, f_exp, intervals_exp, 36, 30)

        edge_id = 9
        intervals_exp = np.array([ts.nodes_time[u] for u in range(20, 25)])
        intervals_exp = np.insert(intervals_exp, 0, 0)
        f_exp = np.array([13, 12, 10, 8, 6])
        self.verify_edge(ts, edge_id, f_exp, intervals_exp, 24, 13)

    def verify_edge(self, ts, edge_id, f_exp, intervals_exp, parent, child):
        tables = ts.tables
        edge = tables.edges[edge_id]
        assert edge.child == child
        assert edge.parent == parent
        # assert edge.left == 1
        f, intervals = lik.lineages_to_left_count(edge, ts)
        assert len(f) == len(intervals) - 1
        assert np.all(np.equal(intervals, intervals_exp))
        assert np.array_equal(f, f_exp)

class TestRunSMC:
    def test_compute_lik(self):
        seeds = np.array([12, 24, 36])
        rec_rate = 1e-5
        pop_size = 1000
        coal_rate = 1 / (2 * pop_size)
        for seed in seeds:
            ts = run_smc(rec_rate, pop_size, seed)
            tables = ts.dump_tables()
            ret = lik.log_likelihood(tables, rec_rate, pop_size)
            assert np.exp(ret) > 0
            assert np.exp(ret) < 1

    def test_compute_lik_b(self):
        seeds = np.array([1212, 46654, 84684])
        rec_rate = 1e-4
        pop_size = 100
        coal_rate = 1 / (2 * pop_size)
        for seed in seeds:
            ts = run_smc(rec_rate, pop_size, seed)
            tables = ts.dump_tables()
            ret = lik.log_likelihood(tables, rec_rate, pop_size)
            assert np.exp(ret) > 0
            assert np.exp(ret) < 1

    def test_compute_lik_seq(self):
        seeds = [12, 23423, 231, 967893]
        rec_rate = 1e-5
        pop_size = 1000
        coal_rate = 1 / (2 * pop_size)
        for seed in seeds:
            ts = run_smc(rec_rate, pop_size, seed)
            tables = ts.dump_tables()
            exp = lik.log_likelihood(tables, rec_rate, pop_size)
            ret = lik.log_likelihood_seq(ts, rec_rate, pop_size)
            assert np.isclose(np.exp(exp), np.exp(ret))
            print(exp, ret)

    def test_compute_lik_seq_simple(self):
        seeds = [3554, 2368, 94720, 836502]
        rec_rate = 5e-6
        pop_size = 1000
        coal_rate = 1 / (2 * pop_size)
        samples = 2
        for seed in seeds:
            ts = run_smc(rec_rate, pop_size, seed, samples)
            tables = ts.dump_tables()
            exp = lik.log_likelihood(tables, rec_rate, pop_size)
            ret = lik.log_likelihood_seq(ts, rec_rate, pop_size)
            assert np.isclose(np.exp(exp), np.exp(ret))
            print(exp, ret)


class TestRunHudson:
    def run_hudson(self, r, pop_size, seed, num_samples=10):
        ts = msprime.sim_ancestry(
            samples=num_samples,
            recombination_rate=r,
            population_size=pop_size,
            sequence_length=100,
            coalescing_segments_only=False,
            random_seed=seed,
            ploidy=2,
        )
        return ts

    def test_compute_lik(self):
        seeds = np.array([6, 36, 72])
        rec_rate = 1e-5
        pop_size = 1000
        coal_rate = 1 / (2 * pop_size)
        for seed in seeds:
            ts = run_hudson(rec_rate, pop_size, seed)
            tables = ts.dump_tables()
            ret = lik.log_likelihood(tables, rec_rate, pop_size)
            assert np.exp(ret) > 0
            assert np.exp(ret) < 1

    def test_no_rec(self):
        seeds = np.array([4544, 146, 2334])
        rec_rate = 0.0
        pop_size = 1000
        coal_rate = 1 / (2 * pop_size)
        num_samples = 2
        for seed in seeds:
            ts = run_hudson(rec_rate, pop_size, seed, num_samples)
            tables = ts.dump_tables()
            ret = lik.log_likelihood(tables, rec_rate, pop_size)
            ret_hudson = msprime.log_arg_likelihood(
                ts, recombination_rate=0.0, Ne=pop_size
            )
            assert np.isclose(ret_hudson, ret)

    def test_no_rec_seq(self):
        seeds = np.array([4544, 146, 2334])
        rec_rate = 0.0
        pop_size = 1000
        coal_rate = 1 / (2 * pop_size)
        num_samples = 2
        for seed in seeds:
            ts = run_hudson(rec_rate, pop_size, seed, num_samples)
            print(ts.draw_text())
            ret = lik.log_likelihood_seq(ts, rec_rate, pop_size)
            ret_hudson = msprime.log_arg_likelihood(
                ts, recombination_rate=0.0, Ne=pop_size
            )
            assert np.isclose(ret_hudson, ret)

    def test_simplified_ts(self):
        seeds = np.array([6, 36, 72])
        rec_rate = 1e-5
        pop_size = 1000
        coal_rate = 1 / (2 * pop_size)
        for seed in seeds:
            ts = run_hudson(rec_rate, pop_size, seed)
            ts = ts.simplify()
            ret = lik.log_likelihood_seq(ts, rec_rate, pop_size)
            assert ret < 0


class TestLogDepth:
    def test_binary_interval(self):
        left_counts = np.array([8])
        intervals = np.array([12.1, 15.9])
        min_parent_time = 14.1
        rec_rate = 1e-5
        coal_rate = 1 / 2e3
        cum_area = left_counts[0] * (intervals[-1] - intervals[0])

        tmin = min(intervals[-1], min_parent_time)
        exp0 = -rec_rate * intervals[0] - coal_rate * cum_area
        cum_area -= left_counts[0] * (tmin - intervals[0])
        exp1 = -rec_rate * tmin - coal_rate * cum_area
        exp_value = (
            rec_rate
            / (rec_rate - coal_rate * left_counts[0])
            * (np.exp(exp0) - np.exp(exp1))
        )

        obs_value = lik.log_depth(
            min_parent_time,
            left_counts,
            intervals,
            rec_rate,
            coal_rate,
            True,
        )
        assert np.isclose(obs_value, np.log(exp_value))

    def test_triple_interval(self):
        left_counts = np.array([8, 7])
        intervals = np.array([12.1, 13.9, 15.9])
        interval_lengths = intervals[1:] - intervals[:-1]
        min_parent_time = 14.1
        rec_rate = 1e-5
        coal_rate = 1 / 2e3
        cum_area = np.sum(left_counts * interval_lengths)

        def _r(f):
            return rec_rate / (rec_rate - coal_rate * f)

        ret = _r(left_counts[0]) * np.exp(
            -rec_rate * intervals[0] - coal_rate * cum_area
        )
        cum_area -= 1.8 * 8
        ret += (_r(left_counts[1]) - _r(left_counts[0])) * np.exp(
            -rec_rate * intervals[1] - coal_rate * cum_area
        )
        cum_area -= 0.2 * 7
        ret -= _r(left_counts[1]) * np.exp(
            -rec_rate * min_parent_time - coal_rate * cum_area
        )

        obs_value = lik.log_depth(
            min_parent_time,
            left_counts,
            intervals,
            rec_rate,
            coal_rate,
            True,
        )
        assert np.isclose(obs_value, np.log(ret))


class TestNumba:
    def test_nodes_time(self):
        rec_rate = 5e-6
        pop_size = 1000
        coal_rate = 1 / (2 * pop_size)
        seed = 2592
        assert True

    def test_binary_search(self):
        x = np.array([0.1, 0.5, 1.2, 1.6, 1.8, 2.0, 3.0, 9.0])
        ret = liknb.binary_search(x, 1.8, 0, 4)
        assert ret == 4
        ret = liknb.binary_search(x, 1.8, 4, x.size)
        assert ret == 4

    def test_counts(self):
        intervals = np.array([0, 2, 4, 6, 8], np.float64)
        counts = np.array([4, 3, 2, 1, 1], np.int64)
        i = liknb.update_counts_descending(intervals, counts, 4, 2, -1)
        assert i == 0
        exp_counts = np.array([4, 2, 1, 0, 1], dtype=np.int64)
        assert np.array_equal(counts, exp_counts)

        i = liknb.update_counts_descending(intervals, counts, 4, 0, +1)
        assert i == -1
        exp_counts = np.array([5, 3, 2, 1, 1], dtype=np.int64)
        assert np.array_equal(counts, exp_counts)

    def test_triple_interval(self):
        left_counts = np.array([8, 7])
        intervals = np.array([12.1, 13.9, 15.9])
        interval_lengths = intervals[1:] - intervals[:-1]
        min_parent_time = 14.1
        rec_rate = 1e-5
        coal_rate = 1 / 2e3
        cum_area = np.sum(left_counts * interval_lengths)

        def _r(f):
            return rec_rate / (rec_rate - coal_rate * f)

        ret = _r(left_counts[0]) * np.exp(
            -rec_rate * intervals[0] - coal_rate * cum_area
        )
        cum_area -= (intervals[1] - intervals[0]) * 8
        ret += (_r(left_counts[1]) - _r(left_counts[0])) * np.exp(
            -rec_rate * intervals[1] - coal_rate * cum_area
        )
        cum_area -= (min_parent_time - intervals[1]) * 7
        ret -= _r(left_counts[1]) * np.exp(
            -rec_rate * min_parent_time - coal_rate * cum_area
        )

        obs_value = liknb.log_depth(
            min_parent_time,
            left_counts,
            intervals,
            rec_rate,
            coal_rate,
            True,
        )
        ret = np.log(ret)
        assert np.isclose(obs_value, ret)
        left_counts = np.array([8, 7, 7])
        intervals = np.array([12.1, 13.9, 14.1, 15.9])
        parent_ptr = 3
        obs_value = liknb.log_depth_descending(
            left_counts,
            intervals,
            intervals[-2],
            intervals[0],
            parent_ptr,
            rec_rate,
            coal_rate,
            True,
        )
        assert np.isclose(obs_value, ret)

    def test_no_rec(self):
        seeds = np.array([4544, 146, 2334])
        rec_rate = 0.0
        pop_size = 1000
        coal_rate = 1 / (2 * pop_size)
        num_samples = 2
        for seed in seeds:
            ts = run_hudson(rec_rate, pop_size, seed, num_samples)
            ret = liknb.log_likelihood(ts, rec_rate, pop_size)
            ret_hudson = msprime.log_arg_likelihood(
                ts, recombination_rate=0.0, Ne=pop_size
            )
            ret2 = liknb.log_likelihood_descending(ts, rec_rate, pop_size)
            assert np.isclose(ret_hudson, ret)
            assert np.isclose(ret2, ret)

    def test_compute_lik_seq_simple(self):
        seeds = [3554, 2368, 94720, 836502]
        rec_rate = 5e-6
        pop_size = 1000
        coal_rate = 1 / (2 * pop_size)
        samples = 2
        for seed in seeds:
            ts = run_smc(rec_rate, pop_size, seed, samples)
            exp = lik.log_likelihood_seq(ts, rec_rate, pop_size)
            ret = liknb.log_likelihood(ts, rec_rate, pop_size)
            ret2 = liknb.log_likelihood_descending(ts, rec_rate, pop_size)
            assert np.isclose(exp, ret)
            print(ret2, ret, exp, seed)
            assert np.isclose(ret2, ret)

    def test_compute_lik_seq(self):
        seeds = [12, 23423, 231, 967893]
        rec_rate = 1e-5
        pop_size = 1000
        coal_rate = 1 / (2 * pop_size)
        for seed in seeds:
            ts = run_smc(rec_rate, pop_size, seed)
            exp = lik.log_likelihood_seq(ts, rec_rate, pop_size)
            ret = liknb.log_likelihood(ts, rec_rate, pop_size)
            ret2 = liknb.log_likelihood_descending(ts, rec_rate, pop_size)
            ret3 = liknb.log_likelihood_descending_numba(ts, rec_rate, pop_size)
            print(ret3, ret2, ret, exp)
            assert np.isclose(exp, ret)
            assert np.isclose(ret, ret2)
            assert np.isclose(ret, ret3)
