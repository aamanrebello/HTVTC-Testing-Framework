#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

import optuna
from sklearn import svm
from commonfunctions import generate_range
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics
import time

#Library only applicable in linux
#from resource import getrusage, RUSAGE_SELF

task = 'classification'
data = loadData(source='sklearn', identifier='breast_cancer', task=task)
data_split = trainTestSplit(data)
train_X = data_split['training_features']
train_y = data_split['training_labels']
validation_X = data_split['validation_features']
validation_y = data_split['validation_labels']

metric=classificationmetrics.indicatorFunction
resolution = 0.2


def obtain_hyperparameters(trial):
    C = trial.suggest_float("C", 0.05, 5.05, step=0.05)
    gamma = trial.suggest_float("gamma", 0.05, 5.05, step=0.05)
    return C, gamma


def objective(trial):
    C, gamma = obtain_hyperparameters(trial)
    training_size = len(train_X)

    metric_value = None

    for fraction in generate_range(resolution,1,resolution):
        #Generate fraction of training data
        trial_size = int(fraction*training_size)
        trial_train_X = train_X[:trial_size]
        trial_train_y = train_y[:trial_size]
        #Train SVM with hyperparameters on data
        clf = svm.SVC(C = C, kernel = 'rbf', gamma = gamma)
        #Make prediction
        clf.fit(trial_train_X, trial_train_y)
        trial_validation_predictions = clf.predict(validation_X)
        metric_value = metric(validation_y, trial_validation_predictions)
        #Check for pruning
        trial.report(metric_value, fraction)
        if trial.should_prune():
            print('=======================================================================================================')
            raise optuna.TrialPruned()

    #Would return the metric for fully trained model (on full dataset)
    return metric_value
    

start_time = time.perf_counter()

study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=resolution, max_resource=1, reduction_factor=2
    ),
)
study.optimize(objective, n_trials=500)
#resource_usage = getrusage(RUSAGE_SELF)
end_time = time.perf_counter()
print('\n\n\n')
print(f'Best trial: {study.best_trial}')
print(f'Execution time: {end_time - start_time}')
#print(f'Resource usage: {resource_usage}')
