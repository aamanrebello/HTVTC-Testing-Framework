import math

# Raises exception if lists have different lengths
def ensureEqualLength(leftList, rightList):
    if len(leftList) != len(rightList):
        raise ValueError('Input lists need to be of same length.')

# Indicator Function
def indicatorFunction(predictions, true_values, **kwargs):
    ensureEqualLength(predictions, true_values)

    def pairIndicator(prediction, true):
        if prediction != 1 and prediction != 0:
            raise ValueError('The predictions must either be 0 or 1.')
        if true != 1 and true != 0:
            raise ValueError('The true values must either be 0 or 1.')
        return int(prediction != true)

    indicator_losses = list(map(pairIndicator, predictions, true_values))
    return sum(indicator_losses) / len (indicator_losses)

# Hinge loss
def hingeLoss(predictions, true_values, **kwargs):
    ensureEqualLength(predictions, true_values)
    hinge = lambda a, b : max(1 - a*b, 0)
    hinge_losses = list(map(hinge, predictions, true_values))
    return sum(hinge_losses) / len (hinge_losses)


# Binary cross-entropy
def binaryCrossEntropy(predictions, true_values, **kwargs):
    ensureEqualLength(predictions, true_values)

    def bceTerm(prediction, true):
        tolerance = 1e-10 #To avoid log(0) terms which are undefined
        if prediction > 1 or true > 1:
            raise ValueError('The prediction and true values all need to be between 0 and 1.')
        if prediction < 0 or true < 0:
            raise ValueError('The prediction and true values all need to be between 0 and 1.')
        return -(true*math.log(prediction + tolerance) + (1 - true)*math.log(1 - prediction + tolerance))

    bce_losses = list(map(bceTerm, predictions, true_values))
    return sum(bce_losses) / len(bce_losses)


# Kullback-Leibler divergence
def KullbackLeiblerDivergence(predictions, true_values, **kwargs):
    ensureEqualLength(predictions, true_values)

    def KLTerm(prediction, true):
        tolerance = 1e-10 #To avoid log(0) terms which are undefined
        if prediction > 1 or true > 1:
            raise ValueError('The prediction and true values all need to be between 0 and 1.')
        if prediction < 0 or true < 0:
            raise ValueError('The prediction and true values all need to be between 0 and 1.')
        return true*math.log(true+tolerance) - true*math.log(prediction+tolerance)

    KL_losses = list(map(KLTerm, predictions, true_values))
    return sum(KL_losses) / len(KL_losses)
