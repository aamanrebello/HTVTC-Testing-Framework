taskTypeMap = {
    'c': 'classification',
    'ic': 'imageclassification',
    'tc': 'textclassification',
    'r': 'regression'
}

datasetTaskMap = {
    'iris': 'c',
    'diabetes': 'r',
    'digits': 'ic',
    'wine': 'c',
    'breast_cancer': 'c',
    'california_housing': 'r',
    'rcv1': 'tc',
    'covtype': 'c'
}

DOWNLOAD_PATH = './datasets'
ONLINE_DOWNLOAD_PATH = DOWNLOAD_PATH + '/online_data.csv'

def generateReturnDict(data, labels):
    return {
        'data': data,
        'labels': labels,
    }

def manipulateLocalData(file_path=ONLINE_DOWNLOAD_PATH, feature_attributes=[], label_attributes=[]):
    import pandas as pd
    df = pd.read_csv(file_path)
    return generateReturnDict(df[feature_attributes].to_numpy(), df[label_attributes].to_numpy())


def loadData(source, identifier, task='classification', **kwargs):
    local_file_path = None

    if source == 'sklearn':
        if identifier in datasetTaskMap.keys():
            if taskTypeMap[datasetTaskMap[identifier]] != task:
                raise ValueError('The specified sklearn dataset does not fit the specified machine learning task.')
        else:
            raise ValueError('The specified sklearn dataset is not recognised.')
        import sklearn.datasets as dat
        data = None
        if identifier == 'iris':
            data = dat.load_iris()
        elif identifier == 'diabetes':
            data = dat.load_diabetes()
        elif identifier == 'digits':
            data = dat.load_digits()
        elif identifier == 'wine':
            data = dat.load_wine()
        elif identifier == 'breast_cancer':
            data = dat.load_breast_cancer()
        elif identifier == 'california_housing':
            data = dat.fetch_california_housing(data_home=DOWNLOAD_PATH)
        elif identifier == 'rcv1':
            data = dat.fetch_rcv1(data_home=DOWNLOAD_PATH)
        elif identifier == 'covtype':
            data = dat.fetch_covtype(data_home=DOWNLOAD_PATH)
        # Generate dictionary in required format
        return generateReturnDict(data['data'], data['target'])

    elif source == 'online':
        import requests
        # Load file from online
        req = requests.get(identifier)
        url_content = req.content
        # Save locally into online download path
        csv_file = open(ONLINE_DOWNLOAD_PATH, 'wb+')
        csv_file.write(url_content)
        csv_file.close()
        # Specify the local data file path as the place where the file was downloaded from online.
        local_file_path = ONLINE_DOWNLOAD_PATH

    elif source == 'local':
        # Specify the local data file path as the path provided as parameter.
        local_file_path = identifier

    else:
        raise ValueError('The specified source of data is not recognised.')

    # If the function has not already returned, this means the data is in a local file and
    # needs to be manipulated into the desired format.

    # The kwargs help decide what aspects of the data are features and labels.
    if 'feature_attributes' in kwargs.keys() and 'label_attributes' in kwargs.keys():
        return manipulateLocalData(local_file_path, kwargs['feature_attributes'], kwargs['label_attributes'])
    else:
        raise ValueError('The feature and label attributes of the data need to be specified.')


def trainTestSplit(data_dictionary, method='separate', test_proportion=0.2, validation_proportion=0.2):

    from sklearn.model_selection import train_test_split
    # Generate test set and joint test and validation set.
    split1 = train_test_split(data_dictionary['data'], data_dictionary['labels'], test_size=test_proportion)
    X_train_and_validation, X_test, y_train_and_validation, y_test = split1

    if method == 'separate':
        # Generate training and validation set.
        split2 = None
        if validation_proportion <= 0:
            split2 = X_train_and_validation, [], y_train_and_validation, []
            X_train, X_validation, y_train, y_validation = split2
        else:
            rescaled_validation_proportion = validation_proportion/(1 - test_proportion)
            split2 = train_test_split(X_train_and_validation, y_train_and_validation, test_size=rescaled_validation_proportion)
            X_train, X_validation, y_train, y_validation = split2
        return {
            'training_features': X_train,
            'training_labels': y_train,
            'validation_features': X_validation,
            'validation_labels': y_validation,
            'test_features': X_test,
            'test_labels': y_test
        }

    elif method == 'cross-validation':
        if validation_proportion <= 0:
            raise ValueError('Validation data size cannot be zero for cross validation.')
        from sklearn.model_selection import KFold
        # Calculate number of folds from proportion of validation data
        n_splits = int(1/validation_proportion)
        # Create generator object
        kf = KFold(n_splits=n_splits, shuffle=True)
        return {
            'index_generator': kf.split(X_train_and_validation),
            'no_splits': kf.get_n_splits(X_train_and_validation)
        }
    else:
        raise ValueError('The specified method to split the data is not recognised.')

'''
method = 'cross-validation'
data = loadData('local', ONLINE_DOWNLOAD_PATH, feature_attributes=['Close'], label_attributes=['Open'])
split = trainTestSplit(data, method, validation_proportion=0.1)

if method == 'separate':
    for key, value in split.items():
        print(f'{key} size: {len(value)}')

elif method == 'cross-validation':
    print(split['no_splits'])
    i = 0
    for train_index, test_index in split['index_generator']:
        print("TRAIN:", len(train_index), "TEST:", len(test_index))
        i += 1
        if i == 5:
            break
'''
