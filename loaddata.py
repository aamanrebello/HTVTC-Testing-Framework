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
    'rcv1': 'tc'
}

DOWNLOAD_PATH = './datasets'

def generateReturnDict(data, labels):
    return {
        'data': data,
        'labels': labels,
    }

def loadData(source, type, task='classification'):
    if source == 'sklearn':
        import sklearn.datasets as dat
        data = None
        if type == 'iris':
            data = dat.load_iris()
        elif type == 'diabetes':
            data = dat.load_diabetes()
        elif type == 'digits':
            data = dat.load_digits()
        elif type == 'wine':
            data = dat.load_wine()
        elif type == 'breast_cancer':
            data = dat.load_breast_cancer()
        elif type == 'california_housing':
            data = dat.fetch_california_housing(data_home=DOWNLOAD_PATH)
        return generateReturnDict(data['data'], data['target'])
    else:
        pass

print(loadData('sklearn', 'rcv1'))
