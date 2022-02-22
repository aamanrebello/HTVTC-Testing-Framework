import math

# Function that subtracts elements in two lists if they are of the same length
def subtractLists(left_list, right_list):
    if len(left_list) != len(right_list):
        raise ValueError('Input lists need to be of same length to be subtracted.')
    subtract = lambda a, b : a - b
    return list(map(subtract, left_list, right_list))

# Mean absolute error
def mae(predictions, true_values, **kwargs):
    errors = subtractLists(predictions, true_values)
    absolute_errors = list(map(abs, errors))
    return sum(absolute_errors) / len (absolute_errors)

# Mean absolute percentage error
def mape(predictions, true_values, **kwargs):
    SMALL_VALUE = 1e-8
    errors = subtractLists(predictions, true_values)
    absolute_errors = list(map(abs, errors))
    absolute_percentage_errors = [(ae*100)/(y + SMALL_VALUE) for (ae, y) in zip(absolute_errors, true_values)]
    return sum(absolute_percentage_errors) / len (absolute_percentage_errors)

# Mean squared error
def mse(predictions, true_values, **kwargs):
    errors = subtractLists(predictions, true_values)
    squared_errors = list(map(lambda a : a**2, errors))
    return sum(squared_errors) / len (squared_errors)

# Mean squared log error
def msle(predictions, true_values, **kwargs):
    log_transform = lambda a : math.log(1 + a)
    logarithmic_predictions = list(map(log_transform, predictions))
    logarithmic_true_values = list(map(log_transform, true_values))
    logarithmic_errors = subtractLists(logarithmic_predictions, logarithmic_true_values)
    logarithmic_squared_errors = list(map(lambda a : a**2, logarithmic_errors))
    return sum(logarithmic_squared_errors) / len (logarithmic_squared_errors)

# log-cosh error
def logcosh(predictions, true_values, **kwargs):
    errors = subtractLists(predictions, true_values)
    log_cosh_transform = lambda a : math.log(math.cosh(a))
    log_cosh_errors = list(map(log_cosh_transform, errors))
    return sum(log_cosh_errors) / len (log_cosh_errors)

# Huber loss
def huber(predictions, true_values, **kwargs):
    delta = None
    if 'delta' not in kwargs.keys():
        delta = 1.35 # Default value in most implementations
    else:
        delta = kwargs['delta']

    def singleObservationHuber(prediction, true_value):
        abs_diff = abs(prediction - true_value)
        if abs_diff <= delta:
            return 0.5*( abs_diff**2 )
        else:
            return delta*abs_diff - 0.5*( delta**2 )

    huber_losses =  list(map(singleObservationHuber, predictions, true_values))
    return sum(huber_losses) / len (huber_losses)

# Poisson loss
def poisson(predictions, true_values, **kwargs):
    singleObservationPoisson = lambda pred, true : pred - true * math.log(pred)
    poisson_losses =  list(map(singleObservationPoisson, predictions, true_values))
    return sum(poisson_losses) / len (poisson_losses)
