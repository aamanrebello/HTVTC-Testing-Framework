# HTVTC-Testing-Framework

This contains code for a framework that can test different aspects of hyperparameter tuning via tensor completion (HTVTC). Experiments on different machine learning models on univariate regression and binary classification may be found in the [experiments](./experiments) folder. Experiments on traditional hyperparameter optimisation techniques on the same problems can be found in the [traditional-methods](./traditional-methods) folder, each technique being experimented with in a different subfolder.

## Programs That Can Be Run

These contain tests that can be run by the end user

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

These are defined in [`regressionmetrics.py`](./regressionmetrics.py) (univariate regression metrics) and [`classificationmetrics.py](./classificationmetrics.py). Refer to these files for the definitions.

### Univariate Regression Metrics

- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Mean Squared Error (MAPE)
- Mean Squared Logarithmic Error (MSLE)
- *logcosh* loss
- Huber loss with default `delta = 1.35`
- Posson loss.

### Binary Classification Metrics

- Indicator loss
- Binary cross-entropy (BCE)
- Kullback-Leibler divergence (KLD)
- Jensen-Shannon divergence (JSD)

## Non-Standard Python Library Dependencies

- **Used throughout the software:** `numpy`, `pandas`, `scipy`, `sklearn`, `tensorly`.
- **Used in folder [traditional-methods](./traditional-methods)**: `optuna`, `bayesian-optimisation`, `bohb-hpo`.
- **Used in module [loadddata.py](./loaddata.py)**: `requests`.

## Version notes
 - Python version 3.10
 - `tensorly` version 0.7.0
 - `numpy` version 1.22
 - `pandas` version 1.4
 - `sklearn` version 1.1
 - `scipy` version 1.8
 - `optuna` version 2.10
 - `bayesian optimisation` version 1.2
 - `bohb` version 0.5
