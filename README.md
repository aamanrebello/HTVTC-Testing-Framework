# HTVTC-Testing-Framework

This framework tests different aspects of hyperparameter tuning via tensor completion (HTVTC). Experiments on different machine learning models for univariate regression and binary classification may be found in the [experiments](./experiments) folder. Experiments on traditional hyperparameter optimisation techniques on the same problems can be found in the [traditional-methods](./traditional-methods) folder, each technique implemented in a different subfolder.

## Contents

1. [Programs That Can Be Run](#programs-that-can-be-run)
2. [Performance Metrics](#performance-metrics)
3. [Validation Loss Metrics](#validation-loss-metrics)
4. [Traditional Hyperparameter Optimisation Techniques](#traditional-hyperparameter-optimisation-techniques)
5. [Structure of the Framework](#structure-of-the-framework)
6. [Non-Standard Python Library Dependencies](#non-standard-python-library-dependencies)
7. [Version Notes](#version-notes)
8. [Compatibility Issue](#compatibility-issue).

## Programs That Can Be Run

These contain tests that can be run by the end user to evaluate HTVTC and traditional hyperparameter optimisation techniques.

- `*_test.py`: Runs unit tests for the module `*.py`.

- `tensorcompletion_instrumentation.py`: Runs performance measurement tests on large tensors for the tensor completion algorithms.

- `experiments/algo_workspace.py`: Runs correctness tests for HTVTC on saved hyperparameter score tensors for the machine learning algorithm `algo`.

- `experiments/algo_instrumentation.py`: Runs performance tests for HTVTC on the machine learning algorithm `algo` measuring validation loss of the suggested hyperparameter combination and one of execution time (in nanoseconds), CPU utilisation time (in nanoseconds) and maximum memory allocation during runtime (in bytes).

- `traditional-methods/method/algo_workspace.py`: Runs performance tests for traditional hyperparameter optimisation method `method` on machine learning algorithm `algo`, measuring validation loss of the suggested hyperparameter combination and one of execution time (in nanoseconds), CPU utilisation time (in nanoseconds) and maximum memory allocation during runtime (in bytes).

## Performance Metrics

These metrics may be found in different testing modules throughout the framework:

- **Validation Loss**: The measures the prediction loss of the machine learning model generated from the specified machine algorithm with specified hyperparameters on the validation data set. It is measured using one of the [validation loss metrics](#validation-loss-metrics).

- **Norm of Difference**: The norm (square root of sum of squares of elements) of the difference between the predicted tensor (from tensor completion) and the true tensor. In some cases this may be normalised by dividing by the norm of the true tensor.

- **Execution Time**: Execution time of a critical program segment e.g. tensor completion, hyperparameter optimisation. It is measured using the `perf_counter_ns()` function ([link](https://docs.python.org/3/library/time.html#time.perf_counter_ns) to docs) from the Python standard library `time`.

- **CPU Utilisation Time**: The total time spent by the critical program segment executing in user and kernel mode in the CPU core(s) of the computer. If the segment execution is parallelised across cores, this metric may be higher than execution time over the same segment. It is measured using the `process_time_ns()` function ([link](https://docs.python.org/3/library/time.html#time.process_time_ns) to docs) from the Python standard library `time`.

- **Maximum Memory Usage**: The maximum amount of RAM allocated to Python objects during execution of the critical program segment. It is measured in bytes using the `tracemalloc` standard Python library ([link](https://docs.python.org/3/library/tracemalloc.html) to docs).

## Validation Loss Metrics

These are defined in [`regressionmetrics.py`](./regressionmetrics.py) (univariate regression metrics) and [`classificationmetrics.py`](./classificationmetrics.py) (binary classification metrics). Refer to these files for the definitions.

### Univariate Regression Metrics

1. Mean Absolute Error (MAE)
2. Mean Absolute Percentage Error (MAPE)
3. Mean Squared Error (MAPE)
4. Mean Squared Logarithmic Error (MSLE)
5. *logcosh* loss
6. Huber loss with default `delta = 1.35`
7. Posson loss.

### Binary Classification Metrics

1. Indicator loss
2. Binary cross-entropy (BCE)
3. Kullback-Leibler divergence (KLD)
4. Jensen-Shannon divergence (JSD).

## Traditional Hyperparameter Optimisation Techniques

- **Grid Search**: Research paper diescribing the technique [here](https://arxiv.org/pdf/2007.15745.pdf) in part 4.1.2. Implementation: `optuna.samplers.GridSampler` docs [here](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.GridSampler.html).
- **Random Search**: Research paper [here](https://jmlr.org/papers/v13/bergstra12a.html). Implementation: `optuna.samplers.RandomSampler` docs [here](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.RandomSampler.html).
- **BO-TPE**: Implementation: `optuna.samplers.TPESampler` docs [here](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html) that also contain links to research papers describing the technique.
- **CMA-ES**: Implementation `optuna.samplers.CmaEsSampler` docs [here](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.CmaEsSampler.html) that also contains links to research papers describing the technique.
- **BO-GP**: Implementation library: `bayesian-optimization` repo [here](https://github.com/fmfn/BayesianOptimization) with excellent explanations on the technique as well as links to research papers describing the technique.
- **Hyperband**: Research paper [here](https://www.jmlr.org/papers/volume18/16-558/16-558.pdf). Implementation `optuna.pruners.HyperbandPruner` docs [here](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html).
- **BOHB**: Research paper [here](https://proceedings.mlr.press/v80/falkner18a.html). Implementation library `bohb-hpo` repo [here](https://github.com/goktug97/bohb-hpo).

## Structure of the Framework

<img src="https://user-images.githubusercontent.com/56508438/175316361-56a601cc-f9be-4d72-935a-79f36b3287ca.png" alt="drawing" width="500" height="450"/>

## Non-Standard Python Library Dependencies

- **Used throughout the software:** `numpy`, `pandas`, `scipy`, `sklearn`, `tensorly`.
- **Used in folder [traditional-methods](./traditional-methods)**: `optuna`, `bayesian-optimization`, `bohb-hpo`.
- **Used in module [loadddata.py](./loaddata.py)**: `requests`.

Take note of the [compatibility issue](compatibility-issue) between `bayesian-optimization` and `scipy`.

## Version Notes
 - Python version 3.10
 - `tensorly` version 0.7.0
 - `numpy` version 1.22
 - `pandas` version 1.4
 - `sklearn` version 1.1
 - `scipy` version 1.8
 - `optuna` version 2.10
 - `bayesian-optimization` version 1.2
 - `bohb-hpo` version 0.5.

## Compatibility Issue

The following [issue](https://github.com/fmfn/BayesianOptimization/issues/300) describes a compatibility issue between the versions of `bayesian-optimization` and `scipy` described in [version notes](#version-notes). The solution is described in [here](https://github.com/fmfn/BayesianOptimization/pull/303) - it is a simple change that can be made to the `bayesian-optimization` library.

Alternatively, the `bayesian-optimization` library can be downloaded as: `pip install git+https://github.com/fmfn/BayesianOptimization`.
