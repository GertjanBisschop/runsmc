Tree-sequence based computation under the Recombination (time) UNaware SMC. 

See the [biorxiv preprint](https://www.biorxiv.org/content/10.1101/2025.02.24.639977v1) for details.

# Documentation

This repository provides fast, flexible implementations for computing the **log-likelihood** of genealogical tree sequences under a coalescent-with-recombination model, using [`tskit`](https://tskit.dev) data structures and accelerated by [`numba`](https://numba.pydata.org/).

---

## Overview

There are **three core likelihood functions**, each of which computes the log-likelihood of a tree sequence (`tskit.TreeSequence`):

### 1. `liknb.log_likelihood(ts, rec_rate, population_size)`
Computes the standard coalescent likelihood with **constant effective population size** across the full timespan of the tree sequence.

### 2. `likstepwise.log_likelihood_stepwise_ne(ts, rec_rate, ne_steps, time_steps)`
Extends the model to support a **piecewise constant effective population size trajectory**. Population size can vary discretely over time, allowing more realistic demographic modeling.

### 3. `likslice.log_likelihood(ts, rec_rate, population_size, time_slice)`
Computes the likelihood **restricted to a slice of time**, useful for focused analysis over specific epochs in the ancestry. Assumes constant population size within the slice.

Each function ultimately calls a common underlying core (`_log_likelihood`), but accepts different parameters to enable these use cases.


## In detail:

### 1. `liknb.log_likelihood(ts, rec_rate, population_size)`

- **`ts`** (`tskit.TreeSequence`):  
  A tree sequence object encoding the ancestry of a sample of genomes. It must include edge information and node times.

- **`rec_rate`** (`float`):  
  The per-base per-generation recombination rate.

- **`population_size`** (`float`):  
  The effective population size used to compute the coalescence rate (`1 / (2 * population_size)`).

- **`rec_correction`** (`bool`, optional, default=`False`):  
  If `True`, enables correction for recurrent recombination events at the same node. This affects how recombination likelihoods are computed when a child node has multiple parents across the genome.

---
#### Returns

- **`float`**:  
  The log-likelihood of the tree sequence.

---


### 2. `likstepwise.log_likelihood_stepwise_ne(ts, rec_rate, ne_steps, time_steps, ploidy=2)`

- **`ts`** (`tskit.TreeSequence`):  
  A tree sequence object containing the ancestry of sampled genomes, including node times and edge relationships.

- **`rec_rate`** (`float`):  
  The per-base per-generation recombination rate.

- **`ne_steps`** (`np.ndarray`):  
  An array of effective population sizes for each time interval (must be positive). The length must match `time_steps`.

- **`time_steps`** (`np.ndarray`):  
  An array of increasing time points defining the boundaries of population size intervals. Must be the same size as `ne_steps`.

- **`ploidy`** (`int`, optional, default=`2`):  
  The ploidy level of the organism. Defaults to diploid. Affects coalescence rate calculation via `1 / (ploidy * Ne)`.

---

#### Returns

- **`float`**:  
  The log-likelihood of the tree sequence under the provided stepwise Ne model.

---

### Requirements

- `ne_steps` and `time_steps` must be 1D NumPy arrays of the same length.
- All values in `ne_steps` must be strictly positive.

---

### 3. `likslice.log_likelihood(ts, rec_rate, population_size, time_slice, ploidy=2)`

- **`ts`** (`tskit.TreeSequence`):  
  The input tree sequence containing the genealogical history of a sample of genomes.

- **`rec_rate`** (`float`):  
  The per-base per-generation recombination rate.

- **`population_size`** (`float`):  
  The constant effective population size assumed for the entire time slice. Used to compute the coalescent rate.

- **`time_slice`** (`tuple[float, float]` or array-like):  

---

#### Returns

- **`float`**:  
  The log-likelihood of a slice of the tree sequence.

---

### Requirements

- time_slice[0] < time_slice[1]