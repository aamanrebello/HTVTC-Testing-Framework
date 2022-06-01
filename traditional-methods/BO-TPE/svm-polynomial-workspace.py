import optuna
from optuna.samplers import TPESampler


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return x ** 2


study = optuna.create_study(sampler=TPESampler())
study.optimize(objective, n_trials=10)
