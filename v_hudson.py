import argparse
import dataclasses
import tskit
import msprime
import numpy as np
import pathlib
import sys
import math
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Callable
from typing import List

import runsmc.likelihoods as lik


@dataclasses.dataclass
class ModelComparison:

    num_reps: int
    samples: int
    recombination_rate: float
    population_size: float
    sequence_length: float
    seed: int
    output_dir: str
    discrete_genome: bool = False

    def run(self, f):
        fr = getattr(self, f)
        return fr()

    def run_sims_full_arg(self):
        return msprime.sim_ancestry(
            samples=self.samples,
            recombination_rate=self.recombination_rate,
            population_size=self.population_size,
            sequence_length=self.sequence_length,
            record_full_arg=True,
            random_seed=self.seed,
            discrete_genome=self.discrete_genome,
            num_replicates=self.num_reps,
        )

    def run_sims_unary(self, model):
        return msprime.sim_ancestry(
            samples=self.samples,
            recombination_rate=self.recombination_rate,
            population_size=self.population_size,
            sequence_length=self.sequence_length,
            coalescing_segments_only=False,
            random_seed=self.seed,
            model=model,
            discrete_genome=self.discrete_genome,
            num_replicates=self.num_reps,
        )

    def hudson_smc_hist(self):
        results = np.zeros((2, self.num_reps))
        models = ['hudson', 'smc']
        for m in range(len(models)):
            for i, ts in tqdm(enumerate(self.run_sims_unary(model=models[m])), total=self.num_reps):
                results[m, i] = lik.log_likelihood(
                    ts.tables, self.recombination_rate, self.population_size
                )
            
        filename = self.output_dir / "hudson_smc_hist.png"
        plot_hist(results, filename, models)

    def unary_simplified(self):
        results = np.zeros((2, self.num_reps))
        for i, ts in tqdm(enumerate(self.run_sims_unary()), total=self.num_reps):
            results[0, i] = lik.log_likelihood(
                ts.tables, self.recombination_rate, self.population_size
            )
            ts = ts.simplify()
            results[1, i] = lik.log_likelihood(
                ts.tables, self.recombination_rate, self.population_size
            )
        filename = self.output_dir / "smc_unary_simpl.png"
        plot_unary_simplified(results, filename)

    def v_hudson(self):
        models = ['hudson', 'smc, unary', 'smc, simplified']
        results = np.zeros(((len(models), self.num_reps)), dtype=np.float64)
        for i, ts in tqdm(enumerate(self.run_sims_full_arg()), total=self.num_reps):
            results[0, i] = msprime.log_arg_likelihood(
                ts, self.recombination_rate, self.population_size
            )
            # simplify ts to remove rec nodes and common ancestor events
            node_type = msprime.NodeType.COMMON_ANCESTOR | msprime.NodeType.RECOMBINANT
            retain_nodes = np.bitwise_and(ts.tables.nodes.flags, node_type.value) == 0
            nodes = np.arange(ts.num_nodes)[retain_nodes]
            ts = ts.simplify(
                samples=nodes,
                )
            tables = ts.dump_tables()
            # modify tables to correctly mark sample nodes again
            for j in range(self.samples * 2, ts.num_nodes):
                node_obj = tables.nodes[j]
                node_obj = node_obj.replace(flags=0)
                tables.nodes[j] = node_obj
            ts = tables.tree_sequence()
            results[1, i] = lik.log_likelihood(
                ts.tables, self.recombination_rate, self.population_size
            )
            ts_simpl = ts.simplify()
            results[2, i] = lik.log_likelihood(
                ts_simpl.tables, self.recombination_rate, self.population_size
            )
        results.dump('v_hudson_seed_42.npy')
        filename = self.output_dir / "v_hudson_unary_simpl.png"
        plot_v_hudson(results, filename)

def plot_v_hudson(result, filename):
    mean_plot = np.mean(result)
    min_plot = np.min(result) + 0.1 * mean_plot
    max_plot = np.max(result) - 0.1 * mean_plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(result[0], result[1], marker='o', label='unary')
    ax.scatter(result[0], result[2], marker='x', label='full_simplify')
    ax.set_xlabel('hudson')
    ax.set_ylabel('smc')
    ax.set_title('simulation model: hudson')
    ax.set_xlim((min_plot, max_plot))
    ax.set_ylim((min_plot, max_plot))
    # add legend
    r1 = (np.corrcoef(result[0], result[1])[0, -1])**2
    r2 = (np.corrcoef(result[0], result[2])[0, -1])**2
    ax.annotate(
        'r2_un = {:.2f}, r2_simpl = {:.2f}'.format(r1, r2),
        xy=(0.1, 0.1), 
        xycoords='axes fraction',
    )
    plt.legend(loc="lower right")

    fig.savefig(filename, dpi=70)

def plot_unary_simplified(result, filename):
    mean_plot = np.mean(result)
    min_plot = np.min(result) + 0.1 * mean_plot
    max_plot = np.max(result) - 0.1 * mean_plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(result[0], result[1])
    ax.set_xlabel('smc, unary')
    ax.set_ylabel('smc, simplified')
    ax.set_title('simulation model: smc')
    ax.set_xlim((min_plot, max_plot))
    ax.set_ylim((min_plot, max_plot))

    fig.savefig(filename, dpi=70)

def plot_hist(result, filename, labels):
    mean_plot = np.mean(result)
    min_plot = np.min(result) + 0.1 * mean_plot
    max_plot = np.max(result) - 0.1 * mean_plot
    bins = np.linspace(min_plot, max_plot, num=20)
    fig, ax = plt.subplots(figsize=(10, 10))

    for i in range(result.shape[0]):
        ax.hist(
            result[i], 
            bins, 
            alpha=0.5, 
            density=True, 
            label=labels[i],
            edgecolor='black', 
            linewidth=0.75
        )

    plt.legend(loc='upper right')
    fig.savefig(filename, dpi=70)

def set_output_dir(output_dir, samples, info_str):
    output_dir = pathlib.Path(output_dir + f"/n_{samples}/" + info_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_all(fs, output_dir, seed):
    # parameters
    rho = 1e-8
    L = 1e5
    num_reps = 500
    n = 20
    population_size = 10_000
    
    info_str = f"L_{L}_rho_{rho}"
    output_dir = set_output_dir(output_dir, n, info_str)
    simtracker = ModelComparison(num_reps, n, rho, population_size, L, seed, output_dir)
    for f in fs:
        results = simtracker.run(f)


def main():
    parser = argparse.ArgumentParser()
    choices = [
        "v_hudson",
        "unary_simplified",
        "hudson_smc_hist",
    ]

    parser.add_argument(
        "--functions",
        "-f",
        nargs="*",
        default=choices,
        choices=choices,
        help="Run all the specified functions.",
    )

    parser.add_argument(
        "--output-dir",
        "-d",
        type=str,
        default="_output/v_hudson",
        help="specify the base output directory",
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="specify used seed",
    )

    args = parser.parse_args()

    run_all(args.functions, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
