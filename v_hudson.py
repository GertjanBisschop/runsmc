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
    extract_info: List[Callable[[tskit.TreeSequence], np.float64]]
    seed: int
    discrete_genome: bool = False

    def __post_init__(self):
        self.num_functions = len(self.extract_info)
        self.shape = (self.num_functions, self.num_reps)
        # self.rng = np.random.default_rng(self.seed)

    def run_sims(self):
        return msprime.sim_ancestry(
            samples=self.samples,
            recombination_rate=self.recombination_rate,
            population_size=self.population_size,
            sequence_length=self.sequence_length,
            record_full_arg=True,
            random_seed=self.seed,
            discrete_genome=self.discrete_genome,
            num_replicates = self.num_reps
            )

    def run_models(self, *models):
        assert len(models) > 1, "At least 2 model names are required for a comparison"
        results = np.zeros((len(models), *self.shape), dtype=np.float64)
        for i, ts in tqdm(enumerate(self.run_sims()), total=self.num_reps):
            results[0, 0, i] = msprime.log_arg_likelihood(ts, self.recombination_rate, self.population_size)
            # simplify ts to remove rec nodes and common ancestor events
            ts = ts.simplify()
            results[1, 0, i] = lik.log_likelihood(ts.tables, self.recombination_rate, self.population_size)
        return results

def compare_models_plot(result, models, functions, shape, figsize, filename):
    if len(functions) == 1:
        mean_plot = np.mean(result)
        min_plot = np.min(result) + 0.1 * mean_plot
        max_plot = np.max(result) - 0.1 * mean_plot
        fig, ax = plt.subplots(figsize=(10,10))
        ax.scatter(result[0, 0], result[1, 0])
        ax.set_xlabel(models[0])
        ax.set_ylabel(models[1])
        ax.set_title(functions[0])
        ax.set_xlim((min_plot, max_plot))
        ax.set_ylim((min_plot, max_plot))

    else:
        fig, ax = plt.subplots(*shape, figsize=figsize)
        ax_flat = ax.flat
        for i, function in enumerate(functions):
            ax_flat[i].scatter(result[0, i], result[1, i])
            ax_flat[i].set_xlabel(models[0])
            ax_flat[i].set_ylabel(models[1])
            ax_flat[i].set_title(function)

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
    models = ['hudson', 'smc']

    simtracker = ModelComparison(num_reps, n, rho, population_size, L, fs, seed)
    results = simtracker.run_models(*models)
    info_str = f"L_{L}_rho_{rho}"
    output_dir = set_output_dir(output_dir, n, info_str)
    filename = output_dir / "v_hudson.png"
    subplots_shape = (math.ceil(len(fs) / 2), 2)
    fig_dims = tuple(4 * i for i in subplots_shape[::-1])
    compare_models_plot(
        results,
        models,
        fs,
        subplots_shape,  # shape
        fig_dims,  # size of fig
        filename,
        )

def main():
    parser = argparse.ArgumentParser()
    choices = [
        "likelihood_scatter",
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