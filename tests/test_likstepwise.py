import msprime
import math
import numpy as np
import pytest
import tskit
import mpmath as mp

import runsmc.liknb as liknb
import runsmc.likstepwise as likstep
import runsmc.legacy.likdescending as likdes


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


class TestMergeTimeArrays:
    def test_basic2(self):
        unique_times = np.array([2, 4, 9])
        pop_size_step_times = np.array([0, 3, 4, 11, 13])
        node_map = np.array([0, 0, 0, 1, 1, 2, 2, 2])
        merged, new_node_map, new_rate_map = likstep.merge_time_arrays(
            unique_times, pop_size_step_times, node_map
        )
        assert merged.size == new_rate_map.size
        assert new_node_map.size == node_map.size
        exp_merged = np.unique(np.concatenate([unique_times, pop_size_step_times]))
        assert np.array_equal(exp_merged, merged)
        exp_node_map = np.array([1, 1, 1, 3, 3, 4, 4, 4])
        assert np.array_equal(exp_node_map, new_node_map)
        exp_rate_map = np.array([0, 0, 1, 2, 2, 3, 4])
        assert np.array_equal(exp_rate_map, new_rate_map)

    def test_basic3(self):
        unique_times = np.array([2, 4, 9, 14])
        pop_size_step_times = np.array([0, 3, 4, 11, 13])
        node_map = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3])
        merged, new_node_map, new_rate_map = likstep.merge_time_arrays(
            unique_times, pop_size_step_times, node_map
        )
        assert merged.size == new_rate_map.size
        assert new_node_map.size == node_map.size
        exp_merged = np.unique(np.concatenate([unique_times, pop_size_step_times]))
        assert np.array_equal(exp_merged, merged)
        exp_node_map = np.array([1, 1, 1, 3, 3, 4, 7, 7, 7])
        assert np.array_equal(exp_node_map, new_node_map)
        exp_rate_map = np.array([0, 0, 1, 2, 2, 3, 4, 4])
        assert np.array_equal(exp_rate_map, new_rate_map)

    def test_basic4(self):
        unique_times = np.array([0, 2, 4, 9, 14])
        pop_size_step_times = np.array([0, 3, 4, 11, 13])
        node_map = np.array([0, 0, 0, 1, 1, 2, 3, 3, 4])
        merged, new_node_map, new_rate_map = likstep.merge_time_arrays(
            unique_times, pop_size_step_times, node_map
        )
        assert merged.size == new_rate_map.size
        assert new_node_map.size == node_map.size
        exp_merged = np.unique(np.concatenate([unique_times, pop_size_step_times]))
        assert np.array_equal(exp_merged, merged)
        exp_node_map = np.array([0, 0, 0, 1, 1, 3, 4, 4, 7])
        assert np.array_equal(exp_node_map, new_node_map)
        exp_rate_map = np.array([0, 0, 1, 2, 2, 3, 4, 4])
        print(new_rate_map)
        assert np.array_equal(exp_rate_map, new_rate_map)


class TestLogDepth:
    def test_triple_interval(self):
        mp.dps = 50
        left_counts = np.array([8, 7])
        intervals = np.array([12.1, 13.9, 15.9])
        interval_lengths = intervals[1:] - intervals[:-1]
        min_parent_time = 14.1
        rec_rate = 1e-5
        coal_rate = 1 / 2e3
        cum_area = np.sum(left_counts * interval_lengths)

        def _r(f):
            return rec_rate / (rec_rate - coal_rate * f)

        temp = np.zeros(3)
        ret = _r(left_counts[0]) * mp.exp(
            -rec_rate * intervals[0] - coal_rate * cum_area
        )
        temp[0] = ret
        cum_area -= (intervals[1] - intervals[0]) * 8
        temp[1] = (_r(left_counts[1]) - _r(left_counts[0])) * mp.exp(
            -rec_rate * intervals[1] - coal_rate * cum_area
        )

        cum_area -= (min_parent_time - intervals[1]) * 7
        temp[-1] = -_r(left_counts[1]) * mp.exp(
            -rec_rate * min_parent_time - coal_rate * cum_area
        )
        ret = np.log(np.sum(temp))
        # using log_depth_descending
        left_counts = np.array([8, 7, 7])
        intervals = np.array([12.1, 13.9, 14.1, 15.9])
        child_ptr = 0
        parent_ptr = 3
        coal_rate_array = np.full(intervals.size, 1 / 2e3)
        obs_value = likstep.log_depth_descending(
            left_counts,
            intervals,
            intervals[-2],
            child_ptr,
            parent_ptr,
            rec_rate,
            coal_rate_array,
            True,
        )
        assert np.isclose(obs_value, ret)


class TestLogLik:
    def test_constant_size(self):
        seeds = [12, 23423, 231, 967893]
        rec_rate = 1e-5
        pop_size = 1000
        coal_rate = 1 / (2 * pop_size)
        for seed in seeds:
            ts = run_smc(rec_rate, pop_size, seed)
            exp = likdes.log_likelihood_descending(ts, rec_rate, pop_size)
            time_steps = (np.arange(10) * 1000).astype(np.float64)
            pop_size_array = np.full(time_steps.size, pop_size)
            ret = likstep.log_likelihood_stepwise_ne(
                ts, rec_rate, pop_size_array, time_steps
            )
            print(exp, ret)
            assert np.isclose(exp, ret)

class TestBeyondRoot:
    def test_compute_lik_seq(self):
        seeds = [12, 23423, 231, 967893]
        rec_rate = 1e-5
        pop_size = np.array([1000])
        time_steps = np.zeros(1)
        coal_rate = 1 / (2 * pop_size)
        for seed in seeds:
            ts = run_smc(rec_rate, pop_size, seed)
            ret = likstep.log_likelihood_stepwise_ne(
                ts, 
                rec_rate,
                pop_size,
                time_steps,
            )
            assert ret < 0

    def test_slice(self):
        seed = 12
        rec_rate = 1e-5
        pop_size = np.array([1000])
        time_steps = np.zeros(1)
        ts = run_smc(rec_rate, pop_size, seed)
        min_root_time = np.min(
            [tree.time(tree.root) for tree in ts.trees()]
        )
        sts = ts.decapitate(min_root_time)
        stss = sts.simplify(keep_unary=True)
        ret = likstep.log_likelihood_stepwise_ne(
            stss, 
            rec_rate,
            pop_size,
            time_steps,
        )
        exp = liknb.log_likelihood_descending_numba(
            stss, rec_rate, pop_size[0]
        )    
        assert np.isclose(exp, ret)
        ret2 = likstep.log_likelihood_stepwise_ne(
            ts, 
            rec_rate,
            pop_size,
            time_steps,
        )
        assert ret2 < ret