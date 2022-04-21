from loaddata import loadData, trainTestSplit
import regressionmetrics
import classificationmetrics

# Returns a function 'evaluate' that accepts hyperparameters for the specified
# machine learning algorithm and evaluates a model trained with these hyperparameters
# on the validation dataset
def evaluationFunctionGenerator(data, algorithm = 'svm-rbf', task='classification'):
    train_X = data['training_features']
    train_y = data['training_labels']
    validation_X = data['validation_features']
    validation_y = data['validation_labels']

    # Ridge regression (1 hyperparameter)
    if algorithm == 'ridgeregression' and task=='regression':
        def evaluate(alpha, metric, **kwargs):
            from sklearn.linear_model import Ridge
            clf = Ridge(alpha = alpha)
            clf.fit(train_X, train_y)
            validation_predictions = clf.predict(validation_X)
            return metric(validation_y, validation_predictions, **kwargs)
        return evaluate

    # SVM using radial basis function kernel (2 hyperparameters)
    elif algorithm == 'svm-rbf' and task=='classification':
        def evaluate(C, gamma, metric, **kwargs):
            from sklearn import svm
            clf = svm.SVC(C = C, kernel = 'rbf', gamma = gamma)
            clf.fit(train_X, train_y)
            validation_predictions = clf.predict(validation_X)
            return metric(validation_y, validation_predictions, **kwargs)
        return evaluate

    # SVM using polynomial kernel (4 hyperparameters)
    elif algorithm == 'svm-polynomial' and task=='classification':
        def evaluate(C, gamma, constant_term, degree, metric, **kwargs):
            from sklearn import svm
            clf = svm.SVC(C = C, kernel = 'poly', gamma = gamma, degree = degree, coef0 = constant_term)
            clf.fit(train_X, train_y)
            validation_predictions = clf.predict(validation_X)
            return metric(validation_y, validation_predictions, **kwargs)
        return evaluate

    # K-nearest neighbour regression (3 hyperparameters)    
    elif algorithm == 'knn-regression' and task == 'regression':
        def evaluate(N, weightingFunction, distanceFunction, metric, **kwargs):
            from sklearn.neighbors import KNeighborsRegressor
            clf = None
            if distanceFunction == 'minkowski': # Stands for generalised Minkowski distance
                p = None
                if 'p' not in kwargs.keys():
                    p = 2 # Use Euclidean distance by default
                else:
                    p = kwargs['p'] # Use provided value of p
                clf = KNeighborsRegressor(n_neighbors=N, weights=weightingFunction, p=p)
            else:
                clf = KNeighborsRegressor(n_neighbors=N, weights=weightingFunction, metric=distanceFunction)
            clf.fit(train_X, train_y)
            validation_predictions = clf.predict(validation_X)
            return metric(validation_y, validation_predictions, **kwargs)
        return evaluate

    # K-nearest neighbour classification (3 hyperparameters)
    elif algorithm == 'knn-classification' and task=='classification':
        def evaluate(N, weightingFunction, distanceFunction, metric, **kwargs):
            from sklearn.neighbors import KNeighborsClassifier
            clf = None
            if distanceFunction == 'minkowski': # Stands for generalised Minkowski distance
                p = None
                if 'p' not in kwargs.keys():
                    p = 2 # Use Euclidean distance by default
                else:
                    p = kwargs['p'] # Use provided value of p
                clf = KNeighborsClassifier(n_neighbors=N, weights=weightingFunction, p=p)
            else:
                clf = KNeighborsClassifier(n_neighbors=N, weights=weightingFunction, metric=distanceFunction)
            clf.fit(train_X, train_y)
            validation_predictions = clf.predict(validation_X)
            return metric(validation_y, validation_predictions, **kwargs)
        return evaluate

    # Random forest classification (6 hyperparameters)
    elif algorithm == 'random-forest' and task=='classification':
        def evaluate(no_trees, max_tree_depth, bootstrap, min_samples_split, no_features, metric, **kwargs):
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=no_trees, max_depth=max_tree_depth, bootstrap=bootstrap, min_samples_split=min_samples_split, max_features=no_features, random_state=0)
            clf.fit(train_X, train_y)
            validation_predictions = clf.predict(validation_X)
            return metric(validation_y, validation_predictions, **kwargs)
        return evaluate

    else:
        raise ValueError('The algorithm specified is not recognised.')


#Let's test the function
task = 'classification'
data = loadData(source='sklearn', identifier='breast_cancer', task=task)
data_split = trainTestSplit(data, validation_proportion=0.3)
func = evaluationFunctionGenerator(data_split, algorithm = 'random-forest', task=task)
print(func(15, 10, True, 2, 5, metric=classificationmetrics.indicatorFunction))
