import msprime
import tskit
import numpy as np

def count_rec_events(ts, reset=False):
    rec_events = 0
    last_parent_array = -np.ones(ts.num_nodes, dtype=np.int64)
    for _, edges_out, edges_in in ts.edge_diffs():
        for edge in edges_out:
            p = edge.parent
            c = edge.child
            last_parent_array[c] = p
            
        for edge in edges_in:
            p = edge.parent
            c = edge.child
            last_parent = last_parent_array[c]
            if last_parent != -1:
                if p != last_parent:
                    rec_events += 1
        if reset:
            for edge in edges_out:
                last_parent_array[edge.child] = -1
    return rec_events

def run_smc_sim(num_replicates, record_unary=True):
    rng = np.random.default_rng()
    for i in range(num_replicates):
        seed = rng.integers(1, 2**16)        
        sim = msprime.ancestry._parse_sim_ancestry(
            samples=4,
            recombination_rate=1e-5,
            population_size=1e4,
            sequence_length=100,
            model='smc',
            coalescing_segments_only= not record_unary,
            random_seed=seed,
            ploidy=2
        )
        sim.run()
        assert sim.num_recombination_events > 0
        ts = tskit.TableCollection.fromdict(sim.tables.asdict()).tree_sequence()
        yield (ts, seed, sim.num_recombination_events)

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

class TestRecs:
    def test_num_recs(self):
        num_replicates = 100
        for i, (ts, seed, obs_rec) in enumerate(run_smc_sim(100)):
            inferred_rec = count_rec_events(ts)
            assert obs_rec == inferred_rec

class TestBeyondRoot:
    def test_beyond_root(self):
        rng = np.random.default_rng(9948)
        seeds = rng.integers(1, 2**16, 100)
        found_no_rec = 0
        found_rec = 0
        for i in range(seeds.size):
            ts = run_smc(
                1e-2,
                1,
                seeds[i]
            )
            last_parent_array = -np.ones(ts.num_nodes, dtype=np.int64)
            last_tree_parent_array = -np.ones(ts.num_nodes, dtype=np.int64)
            right_co = np.zeros(ts.num_nodes, dtype=np.int64)    
            first_root_time = np.min(
                [tree.time(tree.root) for tree in ts.trees()]
            )
        
            for _, edges_out, edges_in in ts.edge_diffs():
                for edge in edges_out:
                    last_parent_array[edge.child] = edge.parent
                    last_tree_parent_array[edge.child] = edge.parent
                    right_co[edge.parent] = edge.right
                    
                for edge in edges_in:
                    if last_parent_array[edge.child] != -1:
                        if edge.parent != last_parent_array[edge.child]:
                            if last_tree_parent_array[edge.child]==-1:
                                found_rec += 1
                                assert ts.nodes_time[edge.child] >= first_root_time
                        else:
                            found_no_rec += 1
                            assert ts.nodes_time[edge.child] >= first_root_time
                      
                for edge in edges_out:
                    last_tree_parent_array[edge.child] = -1
 
        assert found_no_rec and found_rec