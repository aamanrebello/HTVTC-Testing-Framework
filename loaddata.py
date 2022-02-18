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
    if 'training_attributes' in kwargs.keys() and 'test_attributes' in kwargs.keys():
        return manipulateLocalData(local_file_path, kwargs['training_attributes'], kwargs['test_attributes'])
    else:
        raise ValueError('The feature and label attributes of the data need to be specified.')





print(loadData('local', 'path', training_attributes=[]))
