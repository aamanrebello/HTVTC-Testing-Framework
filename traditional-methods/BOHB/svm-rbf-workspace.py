#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

from bohb import BOHB
import bohb.configspace as cs
from sklearn import svm
from commonfunctions import generate_range
from trainmodels import evaluationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics
import time


task = 'classification'
data = loadData(source='sklearn', identifier='breast_cancer', task=task)
data_split = trainTestSplit(data)
train_X = data_split['training_features']
train_y = data_split['training_labels']
validation_X = data_split['validation_features']
validation_y = data_split['validation_labels']

metric = classificationmetrics.indicatorFunction
MAXVAL = 10

def objective(fraction, C, gamma):
    training_size = len(train_X)
    #Generate fraction of training data
    trial_size = int(fraction*training_size)
    trial_train_X = train_X[:trial_size]
    trial_train_y = train_y[:trial_size]
    #Train SVM with hyperparameters on data
    clf = svm.SVC(C = C, kernel = 'rbf', gamma = gamma)
    #Make prediction
    clf.fit(trial_train_X, trial_train_y)
    trial_validation_predictions = clf.predict(validation_X)
    return metric(validation_y, trial_validation_predictions)
    
def evaluate(params, n_iterations):
    fraction = n_iterations/MAXVAL
    return objective(**params, fraction=fraction)


if __name__ == '__main__':
    C = cs.UniformHyperparameter('C', 0.05, 5.0)
    gamma = cs.UniformHyperparameter('gamma', 0.05, 5.0)
    configspace = cs.ConfigurationSpace([C, gamma])

    opt = BOHB(configspace, evaluate, max_budget=MAXVAL, min_budget=10, eta=2)

    logs = opt.optimize()
    print(logs)
