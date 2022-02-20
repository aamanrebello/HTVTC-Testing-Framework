from loaddata import loadData, trainTestSplit

# Returns a function 'evaluate' that accepts hyperparameters for the specified
# machine learning algorithm and evaluates a model trained with these hyperparameters
# on the validation dataset
def evaluationFunctionGenerator(algorithm = 'ridgeregression', data):
    train_X = data['training_features']
    train_y = data['training_labels']
    validation_X = data['validation_features']
    validation_y = data['validation_labels']

    if algorithm == 'ridgeregression':
        def evaluate(alpha, metric):
            from sklearn.linear_model import Ridge
            clf = Ridge(alpha = alpha)
            clf.fit(train_X, train_y)
            validation_predictions = clf.predict(validation_X)
            return metric(validation_y, validation_predictions)
        return evaluate

    elif algorithm == 'svm-rbf':
        def evaluate(C, gamma, metric):
            from sklearn import svm
            clf = svm.SVC(C = C, kernel = 'rbf', gamma = gamma)
            clf.fit(train_X, train_y)
            validation_predictions = clf.predict(validation_X)
            return metric(validation_y, validation_predictions)
        return evaluate

    elif algorithm == 'svm-polynomial':
        def evaluate(C, gamma, constant_term, degree, metric):
            from sklearn import svm
            clf = svm.SVC(C = C, kernel = 'poly', gamma = gamma, degree = degree, coef0 = constant_term)
            clf.fit(train_X, train_y)
            validation_predictions = clf.predict(validation_X)
            return metric(validation_y, validation_predictions)
        return evaluate

    elif algorithm == 'knn-regression':
        pass

    elif algorithm == 'knn-classification':
        pass

    elif algorithm == 'random-forest':
        pass

    else
        raise ValueError('The algorithm specified is not recognised.')
