import optuna
from optuna.samplers import RandomSampler


def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    return x ** 2


study = optuna.create_study(sampler=RandomSampler())
study.optimize(objective, n_trials=10)
