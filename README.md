# HTVTC-Testing-Framework

This contains code for a framework that can test the tensor completion technique. Experiments on different machine learning models may be found in the [experiments](./experiments) folder. Experiments on traditional hyperparameter optimisation techniques can be found in the [traditional-methods](./traditional-methods) folder, each technique being experimented with in a diffeent subfolder.

## Non-Standard Python Library Dependencies

- **Used throughout the software:** `numpy`, `pandas`, `scipy`, `sklearn`, `tensorly`.
- **Used in folder [traditional-methods](./traditional-methods)**: `optuna`, `bayesian-optimisation`, `bohb-hpo`.
- **Used in module [loadddata.py](./loaddata.py)**: `requests`.

## Version notes
 - Python version 3.10
- `numpy` version 1.22
