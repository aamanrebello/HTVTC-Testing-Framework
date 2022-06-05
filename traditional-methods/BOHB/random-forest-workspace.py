#Enable importing code from parent directory
import os, sys
current_path = os.getcwd()
parent = os.path.dirname(current_path)
sys.path.insert(1, parent)
parent_of_parent = os.path.dirname(parent)
sys.path.insert(1, parent_of_parent)

from bohb import BOHB
import bohb.configspace as cs
from sklearn.ensemble import RandomForestClassifier
from commonfunctions import generate_range
from trainmodels import evaluationFunctionGenerator
from loaddata import loadData, trainTestSplit, extractZeroOneClasses, convertZeroOne
import regressionmetrics
import classificationmetrics
import time


task = 'classification'
data = loadData(source='sklearn', identifier='wine', task=task)
binary_data = extractZeroOneClasses(data)
data_split = trainTestSplit(binary_data)
train_X = data_split['training_features']
train_y = data_split['training_labels']
validation_X = data_split['validation_features']
validation_y = data_split['validation_labels']

metric = classificationmetrics.KullbackLeiblerDivergence
MAXVAL = 10

def objective(fraction, no_trees, max_tree_depth, bootstrap, min_samples_split, no_features):
    training_size = len(train_X)
    #Generate fraction of training data
    trial_size = int(fraction*training_size)
    trial_train_X = train_X[:trial_size]
    trial_train_y = train_y[:trial_size]
    #Train model with hyperparameters on data
    clf = RandomForestClassifier(n_estimators=int(no_trees), max_depth=int(max_tree_depth), bootstrap=bootstrap, min_samples_split=int(min_samples_split), max_features=int(no_features), random_state=0)
    #Make prediction
    clf.fit(trial_train_X, trial_train_y)
    trial_validation_predictions = clf.predict(validation_X)
    return metric(validation_y, trial_validation_predictions)
    
def evaluate(params, n_iterations):
    fraction = n_iterations/MAXVAL
    return objective(**params, fraction=fraction)


if __name__ == '__main__':
    no_trees = cs.CategoricalHyperparameter("no_trees", [1,10,20,30,40])
    max_tree_depth = cs.CategoricalHyperparameter("max_tree_depth", [1, 5, 10, 15, 20])
    bootstrap = cs.CategoricalHyperparameter("bootstrap", [True, False])
    min_samples_split = cs.IntegerUniformHyperparameter("min_samples_split", 2, 10)
    no_features = cs.IntegerUniformHyperparameter("no_features", 1, 10)
    
    configspace = cs.ConfigurationSpace([no_trees, max_tree_depth, bootstrap, min_samples_split, no_features])

    opt = BOHB(configspace, evaluate, max_budget=MAXVAL, min_budget=2, eta=2)

    logs = opt.optimize()
    print(logs)
