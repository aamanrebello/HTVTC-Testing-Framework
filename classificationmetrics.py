import math

# Hinge loss
def hingeLoss(predictions, true_values, **kwargs):
    hinge = lambda a, b : max(1 - a*b, 0)
    hinge_losses = list(map(hinge, predictions, true_values))
    return sum(hinge_losses) / len (hinge_losses)


# Binary cross-entropy
def binaryCrossEntropy(predictions, true_values, **kwargs):

    def bceTerm(prediction, true):
        if prediction > 1 or true > 1:
            raise ValueError('The prediction and true values all need to be between 0 and 1.')
        if prediction < 0 or true < 0:
            raise ValueError('The prediction and true values all need to be between 0 and 1.')
        return -(true*math.log(prediction) + (1 - true)*math.log(1 - prediction))

    bce_losses = list(map(bceTerm, predictions, true_values))
    return sum(bce_losses) / len(bce_losses)


# Binary cross-entropy
def KullbackLeiblerDivergence(predictions, true_values, **kwargs):

    def KLTerm(prediction, true):
        if prediction > 1 or true > 1:
            raise ValueError('The prediction and true values all need to be between 0 and 1.')
        if prediction < 0 or true < 0:
            raise ValueError('The prediction and true values all need to be between 0 and 1.')
        return true*math.log(true/prediction)

    KL_losses = list(map(KLTerm, predictions, true_values))
    return sum(KL_losses) / len(KL_losses)
