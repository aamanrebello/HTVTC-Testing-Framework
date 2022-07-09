#Applies computational budget by constraining training data size to a fraction of
#the total number of samples
def applyTrainingSamplesBudget(training_features, training_labels, budget_fraction):
    training_size = len(training_labels)
    budgeted_size = int(budget_fraction*training_size)
    budgeted_training_features = training_features[:budgeted_size]
    budgeted_training_labels = training_labels[:budgeted_size]
    return budgeted_training_features, budgeted_training_labels


#Applies computational budget by constraining number of features in the training data
#to a fraction of the total number of features
def applyTrainingFeaturesBudget(training_features, validation_features, budget_fraction):
    features_size = len(training_features[0])
    budgeted_features = int(budget_fraction*features_size)
    truncate_features = lambda lst : lst[:budgeted_features]
    budgeted_training_features = list(map(truncate_features, training_features))
    budgeted_validation_features = list(map(truncate_features, validation_features))
    return budgeted_training_features, budgeted_validation_features


# Returns a function 'evaluate' that accepts hyperparameters for the specified
# machine learning algorithm and evaluates a model trained with these hyperparameters
# on the validation dataset
def evaluationFunctionGenerator(data, algorithm = 'svm-rbf', task='classification', **outerkwargs):
    train_X = data['training_features']
    train_y = data['training_labels']
    validation_X = data['validation_features']
    validation_y = data['validation_labels']

    if 'budget_type' in outerkwargs.keys():
        if 'budget_fraction' not in outerkwargs.keys():
            raise ValueError('A budget fraction has not been provided.')
        budget_fraction = outerkwargs['budget_fraction']
        if outerkwargs['budget_type'] == 'samples':
            train_X, train_y = applyTrainingSamplesBudget(train_X, train_y, budget_fraction)
        elif outerkwargs['budget_type'] == 'features':
            train_X, validation_X = applyTrainingFeaturesBudget(train_X, validation_X, budget_fraction)

    # Ridge regression (1 hyperparameter)
    if algorithm == 'ridge-regression' and task=='regression':
        
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
            #----------determine evaluation mode---------------
            evaluation_mode = None
            if 'evaluation_mode' not in kwargs.keys():
                evaluation_mode = 'prediction'
            else:
                evaluation_mode = kwargs['evaluation_mode']
            #----------generate predictions based on evaluation mode---------------
            #The default is to use class predictions
            validation_predictions = clf.predict(validation_X)
            #This uses the raw score used to make the prediction i.e. distance from hyperplane
            if evaluation_mode == 'raw-score': 
                validation_predictions = clf.decision_function(validation_X)
            #----------return final metric---------------------
            return metric(validation_y, validation_predictions, **kwargs)
        
        return evaluate


    # SVM using polynomial kernel (4 hyperparameters)
    elif algorithm == 'svm-polynomial' and task=='classification':
        
        def evaluate(C, gamma, constant_term, degree, metric, **kwargs):
            from sklearn import svm
            clf = svm.SVC(C = C, kernel = 'poly', gamma = gamma, degree = degree, coef0 = constant_term)
            clf.fit(train_X, train_y)
            #----------determine evaluation mode---------------
            evaluation_mode = None
            if 'evaluation_mode' not in kwargs.keys():
                evaluation_mode = 'prediction'
            else:
                evaluation_mode = kwargs['evaluation_mode']
            #----------generate predictions based on evaluation mode---------------
            #The default is to use class predictions
            validation_predictions = clf.predict(validation_X)
            #This uses the raw score used to make the prediction i.e. distance from hyperplane
            if evaluation_mode == 'raw-score': 
                validation_predictions = clf.decision_function(validation_X)
            #----------return final metric---------------------
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
                clf = KNeighborsRegressor(n_neighbors=int(N), weights=weightingFunction, p=p)
            else:
                clf = KNeighborsRegressor(n_neighbors=int(N), weights=weightingFunction, metric=distanceFunction)
            clf.fit(train_X, train_y)
            validation_predictions = clf.predict(validation_X)
            return metric(validation_y, validation_predictions, **kwargs)
        
        return evaluate


    # K-nearest neighbour classification (3 hyperparameters)
    elif algorithm == 'knn-classification' and task=='classification':
        
        def evaluate(N, weightingFunction, distanceFunction, metric, **kwargs):
            from sklearn.neighbors import KNeighborsClassifier
            clf = None
            #----------determine distance metric to be used---------------
            if distanceFunction == 'minkowski': # Stands for generalised Minkowski distance
                p = None
                if 'p' not in kwargs.keys():
                    p = 2 # Use Euclidean distance by default
                else:
                    p = kwargs['p'] # Use provided value of p
                clf = KNeighborsClassifier(n_neighbors=int(N), weights=weightingFunction, p=p)
            else:
                clf = KNeighborsClassifier(n_neighbors=int(N), weights=weightingFunction, metric=distanceFunction)
            clf.fit(train_X, train_y)
            #----------determine evaluation mode---------------
            evaluation_mode = None
            if 'evaluation_mode' not in kwargs.keys():
                evaluation_mode = 'prediction'
            else:
                evaluation_mode = kwargs['evaluation_mode']
            #----------generate predictions based on evaluation mode---------------
            #The default is to use class predictions
            validation_predictions = clf.predict(validation_X)
            #This uses the probability that the sample is from class 1
            if evaluation_mode == 'probability':
                extract_at_index_1 = lambda a : a[1]
                validation_predictions = list(map(extract_at_index_1, clf.predict_proba(validation_X)))
            #----------return final metric---------------------
            return metric(validation_y, validation_predictions, **kwargs)
        
        return evaluate


    # Random forest classification (5 hyperparameters)
    elif algorithm == 'random-forest' and task=='classification':
        
        def evaluate(no_trees, max_tree_depth, bootstrap, min_samples_split, no_features, metric, **kwargs):
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=int(no_trees), max_depth=int(max_tree_depth), bootstrap=bootstrap, min_samples_split=int(min_samples_split), max_features=int(no_features), random_state=0)
            clf.fit(train_X, train_y)
            #----------determine evaluation mode---------------
            evaluation_mode = None
            if 'evaluation_mode' not in kwargs.keys():
                evaluation_mode = 'prediction'
            else:
                evaluation_mode = kwargs['evaluation_mode']
            #----------generate predictions based on evaluation mode---------------
            #The default is to use class predictions
            validation_predictions = clf.predict(validation_X)
            #This uses the probability that the sample is from class 1
            if evaluation_mode == 'probability':
                extract_at_index_1 = lambda a : a[1]
                validation_predictions = list(map(extract_at_index_1, clf.predict_proba(validation_X)))
            #----------return final metric---------------------
            return metric(validation_y, validation_predictions, **kwargs)
        
        return evaluate

    else:
        raise ValueError('The algorithm specified is not recognised.')
