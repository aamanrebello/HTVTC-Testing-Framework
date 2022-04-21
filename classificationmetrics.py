import math
from scipy.spatial import distance

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
        # Find probabilities of complementary events
        prediction_c = 1 - prediction
        true_c = 1 - true
        # Return KL divergence
        return true*math.log((true+tolerance)/(prediction+tolerance)) + true_c*math.log((true_c+tolerance)/(prediction_c+tolerance))

    KL_losses = list(map(KLTerm, predictions, true_values))
    return sum(KL_losses) / len(KL_losses)


# Jensen-Shannon divergence
def JensenShannonDivergence(predictions, true_values, **kwargs):
    ensureEqualLength(predictions, true_values)

    def JSTerm(prediction, true):
        if prediction > 1 or true > 1:
            raise ValueError('The prediction and true values all need to be between 0 and 1.')
        if prediction < 0 or true < 0:
            raise ValueError('The prediction and true values all need to be between 0 and 1.')
        # Find probabilities of complementary events
        prediction_c = 1 - prediction
        true_c = 1 - true
        # Create probability distributions
        P = [prediction, prediction_c]
        T = [true, true_c]
        # Return JS divergence (scipy implementation returns square root: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html)
        return distance.jensenshannon(P, T)**2

    JS_losses = list(map(JSTerm, predictions, true_values))
    return sum(JS_losses) / len(JS_losses)
