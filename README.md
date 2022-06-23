# HTVTC-Testing-Framework

This contains code for a framework that can test the tensor completion technique. Experiments on different machine learning models may be found in the [experiments](./experiments) folder. Experiments on traditional hyperparameter optimisation techniques can be found in the [traditional-methods](./traditional-methods) folder, each technique being experimented with in a diffeent subfolder.

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
