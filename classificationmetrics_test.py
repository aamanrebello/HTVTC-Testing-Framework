from math import log
import unittest
from classificationmetrics import ensureEqualLength, indicatorFunction, hingeLoss, binaryCrossEntropy, KullbackLeiblerDivergence, JensenShannonDivergence

class TestEnsureEqualLength(unittest.TestCase):

    def test_unequal_length(self):
        l1 = [1,0,1,1]
        l2 = [0,1]
        test = False
        try:
            result = ensureEqualLength(l1, l2)
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'Input lists need to be of same length.')
            test = True
        self.assertTrue(test)

    def test_equal_length(self):
        l1 = [1,0,1,1]
        l2 = [0,1,0,1]
        test = False
        try:
            result = ensureEqualLength(l1, l2)
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'Input lists need to be of same length.')
            test = True
        self.assertFalse(test)

class TestIndicator(unittest.TestCase):

    def test_fractional_prediction(self):
        l1 = [1,0,0.9,1]
        l2 = [0,1,0,1]
        test = False
        try:
            result = indicatorFunction(l1, l2)
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The predictions must either be 0 or 1.')
            test = True
        self.assertTrue(test)

    def test_large_true_value(self):
        l1 = [1,0,0,1]
        l2 = [0,1,0,11]
        test = False
        try:
            result = indicatorFunction(l1, l2)
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The true values must either be 0 or 1.')
            test = True
        self.assertTrue(test)

    def test_prediction(self):
        l1 = [1,0,0,1,0,1,0,0,1,1]
        l2 = [0,1,0,1,0,0,0,0,0,1]
        expected_result = 0.4
        result = indicatorFunction(l1, l2)
        self.assertAlmostEqual(expected_result, result)


class TestHingeLoss(unittest.TestCase):

    def test_prediction(self):
        l1 = [0.1,0,1,0.5,10,1,0.2,0,1.8,1]
        l2 = [0.1,1,1,0.7,1,0.9,0.2,0,0.1,1.2]
        expected_result = (1 - 0.01 + 1 + 0 + 1 - 0.35 + 0 + 1 - 0.9 + 1 - 0.04 + 1 + 1 - 0.18 + 0)/10
        result = hingeLoss(l1, l2)
        self.assertAlmostEqual(expected_result, result)

class TestBinaryCrossEntropy(unittest.TestCase):

    def test_negative_prediction(self):
        l1 = [0.1,0,1,-0.5,1.0]
        l2 = [0.1,1,1,0.7,1]
        test = False
        try:
            result = binaryCrossEntropy(l1, l2)
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The prediction and true values all need to be between 0 and 1.')
            test = True
        self.assertTrue(test)

    def test_large_true_value(self):
        l1 = [0.1,0,1,0.5,1.0]
        l2 = [0.1,1,1,7,1]
        test = False
        try:
            result = binaryCrossEntropy(l1, l2)
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The prediction and true values all need to be between 0 and 1.')
            test = True
        self.assertTrue(test)

    def test_prediction(self):
        l1 = [0.1, 0.1, 0.9, 0.5, 0.9, 0.45, 0.2, 0.1, 0.8, 0.11]
        l2 = [0,   1,   1,   0.7, 1,   0.9,  0,   0,   0.1, 0.2]
        expected_result = (0.105360516 + 2.30258509 + 0.105360516 + 0.693147181 + 0.105360516 + 0.778440627 + 0.223143551 + 0.105360516 + 1.47080848 + 0.534682036)/10
        result = binaryCrossEntropy(l1, l2)
        self.assertAlmostEqual(expected_result, result)

    def test_zeroes_and_ones(self):
        l1 = [0,   0.1, 0, 0.5, 1, 0.45, 0.2,   1,   0.8, 0.11]
        l2 = [0,   1,   1, 0.7, 1,   0.9,  0,     0,   0.1, 0.2]
        expected_result = (0 + 2.30258509 + 23.0258509 + 0.693147181 + 0 + 0.778440627 + 0.223143551 + 23.0258509 + 1.47080848 + 0.534682036)/10
        result = binaryCrossEntropy(l1, l2)
        self.assertAlmostEqual(expected_result, result)

class TestKullbackLeiblerDivergence(unittest.TestCase):

    def test_negative_true_value(self):
        l1 = [0.1,1,1,0.7,1]
        l2 = [0.1,0,1,-0.1,1.0]
        test = False
        try:
            result = KullbackLeiblerDivergence(l1, l2)
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The prediction and true values all need to be between 0 and 1.')
            test = True
        self.assertTrue(test)

    def test_large_prediction(self):
        l1 = [0.1,1,1,1.1,1]
        l2 = [0.1,0,1,0.5,1.0]
        test = False
        try:
            result = KullbackLeiblerDivergence(l1, l2)
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The prediction and true values all need to be between 0 and 1.')
            test = True
        self.assertTrue(test)

    def test_prediction(self):
        l1 = [0.1, 0.1, 0.9, 0.5, 0.9, 0.45, 0.2,  0.1,  0.8, 0.11]
        l2 = [0.1, 0.7, 0.5, 0.1, 0.8, 0.63, 0.01, 0.15, 0.1, 0.2]
        expected_result = (0.0 + 1.0325534177382862 + 0.5108256237659907 + 0.3680642071684971 + 0.04440300758688234 + 0.06530385821371285 + 0.18100496057056117 + 0.01223511445226829 + 1.1457255029306632 + 0.03427961210451759)/10
        result = KullbackLeiblerDivergence(l1, l2)
        self.assertAlmostEqual(expected_result, result)

    def test_zeroes_and_ones(self):
        l1 = [0,   0.1, 0, 0.5, 1, 0.45, 0.2,   1,   0.8, 0.11]
        l2 = [0,   1,   1, 0.7, 1, 0.9,  0,     0,   0.1, 0.2]
        expected_result = (0.0 + 2.302585092994046 + 23.025850929940457 + 0.08228287850505178 + 0.0 + 0.45335765328010824 + 0.22314355131420976 + 23.025850930040455 + 1.1457255029306632 + 0.03427961210451759)/10
        result = KullbackLeiblerDivergence(l1, l2)
        self.assertAlmostEqual(expected_result, result)


class TestJensenShannonDivergence(unittest.TestCase):

    def test_negative_true_value(self):
        l1 = [0.1,1,1,0.7,1]
        l2 = [0.1,0,1,-0.1,1.0]
        test = False
        try:
            result = JensenShannonDivergence(l1, l2)
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The prediction and true values all need to be between 0 and 1.')
            test = True
        self.assertTrue(test)

    def test_large_prediction(self):
        l1 = [0.1,1,1,1.1,1]
        l2 = [0.1,0,1,0.5,1.0]
        test = False
        try:
            result = JensenShannonDivergence(l1, l2)
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'The prediction and true values all need to be between 0 and 1.')
            test = True
        self.assertTrue(test)

    def test_prediction(self):
        l1 = [0.1, 0.1, 0.9, 0.5, 0.9, 0.45, 0.2,  0.1,  0.8, 0.11]
        l2 = [0.1, 0.7, 0.5, 0.1, 0.8, 0.63, 0.01, 0.15, 0.1, 0.2]
        expected_result = (0.0 + 0.20503802928608553 + 0.10174922507919676 + 0.10174922507919676 + 0.009966389341172874 + 0.01639651126007382 + 0.057730235413083975 + 0.002874130657717279 + 0.2753961152487704 + 0.00782605551441497)/10
        result = JensenShannonDivergence(l1, l2)
        self.assertAlmostEqual(expected_result, result)

    def test_zeroes_and_ones(self):
        l1 = [0,   0.1, 0, 0.5, 1, 0.45, 0.2,   1,   0.8, 0.11]
        l2 = [0,   1,   1, 0.7, 1, 0.9,  0,     0,   0.1, 0.2]
        expected_result = (0.0 + 0.5255973270178643 + 0.6931471805599453 + 0.021005925701837062 + 0.0 + 0.12397013483349642 + 0.07488176162235435 + 0.6931471805599453 + 0.2753961152487704 + 0.00782605551441497)/10
        result = JensenShannonDivergence(l1, l2)
        self.assertAlmostEqual(expected_result, result)


if __name__ == '__main__':
    # Run tests
    unittest.main()
