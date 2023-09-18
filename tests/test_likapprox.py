import msprime
import math
import numpy as np
import pytest
import tskit

import runsmc.likapprox as likapprox
import runsmc.liknb as liknb

class TestFenwick:
    def test_fenwick(self):
        seed = 967893
        rec_rate = 1e-8
        pop_size = 10000
        coal_rate = 1 / (2 * pop_size)
        ts = msprime.sim_ancestry(
            samples=10,
            recombination_rate=rec_rate,
            population_size=pop_size,
            sequence_length=100_000,
            model="smc",
            coalescing_segments_only=False,
            random_seed=seed,
            ploidy=2,
        )
        ret = likapprox.log_likelihood_descending_numba_approx(
            ts, rec_rate, pop_size
        )
        print(ret)
        ret = liknb.log_likelihood_descending_numba(
            ts, rec_rate, pop_size
        )
        print(ret)
        assert False