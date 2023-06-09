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
        intervals_exp = np.array([ts.nodes_time[u] for u in range(30, 37)])
        f, intervals = lik.lineages_to_left_count(edge, ts)
        f_exp = np.array([8, 7, 6, 5, 4, 2])
        assert np.array_equal(f, f_exp)
        assert np.all(np.equal(intervals, intervals_exp))


class TestRunSMC:
    def run_smc(self, r, pop_size, seed):
        ts = msprime.sim_ancestry(
            samples=10,
            recombination_rate=r,
            population_size=pop_size,
            sequence_length=100,
            model="smc",
            coalescing_segments_only=False,
            random_seed=seed,
            ploidy=2,
        )
        return ts

    def test_compute_lik(self):
    	rec_rate = 1e-5
    	pop_size = 1000
    	coal_rate = 2 * pop_size
    	ts = self.run_smc(rec_rate, pop_size, 12)
    	tables = ts.dump_tables()
    	ret = lik.log_lik(tables, rec_rate, coal_rate)
    	print(ret)
    	assert False