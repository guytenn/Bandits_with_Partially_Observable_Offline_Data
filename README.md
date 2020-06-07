# Bandits with Partially Observable Offline Data

This code is submitted as supplementary material for paper:
"Bandits with Partially Observable Offline Data". 

> _**Partially observable offline data can (still) be used to improve online learning**_

**Prerequisites**

Code requires python3 (>=3.5). You'll also need system packages joblib, yaml and standard packages such as numpy, matplotlib.

**A brief description of files:**

* run.py: Contains main run file for executing code with options (see explanation of options further on)
* env.py: Basic environment that is used. Define context distribution, noise distribution, and linear model.
* data.py: Contains DataManger and MbSampler classes, responsible for creating the offline dataset and maintaining statistics of R12.
* OFUL.py: The oful algorithm with linear side information (see Algorithm 1 in paper)
* train.py: Contains the trainer class which can be used to execute jobs. Returns the vector regret.

**run.py commands:**

* N (default value 5): will run OFUL for `10^N` iterations
* d (default value 30): dimension of context and weights
* K (default value 30): number of arms
* n_seeds (default value 5): number of seeds to run for experiment
* seed (default -1): choose specific seed to run (will randomize if not set)
* l (default value 0.1): '\lambda', regularization parameter
* delta (default value 0.01): high probability parameter
* L_values (default value 0 5 10 15 20 25): a list containing the values of L to run
* alpha_values (default value 0.001 0.01): a list containing the values of '\alpha' to run (optimism paramter)
* max_jobs (default value 40): number of processes to run in parallel
* sigma (default value 1): subgaussian parameter of noise
* x_normalization (default 1): norm of 'x' with be divided by this number, such that `||x|| = 1 / x_normalization`
* perturbations: when set, will run oful with offline dataset
* data_sizes (default values 10000 100000 1000000): a list containing sizes of datasets (different run for every dataset)
* calc_r12: when set, will precalculate `R_{12}` to use, such that `M_a` are known in advance
* verbose: print more information during run
  
**How to run:**

* To run with default parameters try

```
python3.8 run.py --perturbations --verbose
```

* To run with your own paramters, here's an example:

```
python3.8 run.py --N 5 --d 30 --K 30 --delta 0.01 --L_values 0 5 10 15 20 25 --alpha_l_values 0.001 0.01 --max_jobs 40 --sigma 1 --x_normalization 1 --perturbations --data_sizes 1000000 --verbose
```