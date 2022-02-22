from regressionmetrics import subtractLists, mae, mse, mape, msle, logcosh, huber, poisson
from math import log, cosh
import numpy as np
import unittest

class TestSubtractLists(unittest.TestCase):

    def test_unequal_length(self):
        l1 = [1,2,3,4]
        l2 = [1,2,3]
        test = False
        try:
            result = subtractLists(l1, l2)
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'Input lists need to be of same length to be subtracted.')
            test = True
        self.assertTrue(test)

    def test_random_lists(self):
        l1 = [1, 4, 5, 6, 2, 3]
        l2 = [2.3, 3.23, 4.111, 3.09, 1, 4]
        result = subtractLists(l1, l2)
        expected_result = [-1.3, 0.77, 0.889, 2.91, 1, -1]
        for i in range(0, len(expected_result)):
            self.assertAlmostEqual(expected_result[i], result[i])

    def test_equal_lists(self):
        l1 = [1, 4, 5, 6, 2, 3, 9.0908989]
        result = subtractLists(l1, l1)
        expected_result = [0]*len(l1)
        for i in range(0, len(expected_result)):
            self.assertAlmostEqual(expected_result[i], result[i])

    def test_negate_list(self):
        l1 = [0]*5
        l2 = [2.3, 3.23, 4.111, 3.09, 1]
        result = subtractLists(l1, l2)
        for i in range(0, len(l2)):
            self.assertAlmostEqual(result[i], -l2[i])


class TestMae(unittest.TestCase):

    def test_random_lists(self):
        predictions = [1.2, 3.4, 7.1, 0.9]
        true_values = [0.4, 5.6, 4.5, 8.7]
        result = mae(predictions, true_values)
        expected_result = (0.8 + 2.2 + 2.6 + 7.8) / 4
        self.assertEqual(result, expected_result)

    def test_l1_norm(self):
        predictions = [1.2, 3.4, -7.1, 0.9, 5.7, -0.76]
        true_values = [0]*6
        result = mae(predictions, true_values)
        expected_result = (1.2 + 3.4 + 7.1 + 0.9 + 5.7 + 0.76) / 6
        self.assertEqual(result, expected_result)

    def test_perfect_accuracy(self):
        true_values = [0.4, 5.6, 4.5, 8.7]
        result = mae(true_values, true_values)
        expected_result = 0
        self.assertEqual(result, expected_result)

    def test_zero_length(self):
        predictions = []
        true_values = []
        test = False
        try:
            result = mae(predictions, true_values)
        except ZeroDivisionError:
            test = True
        self.assertTrue(test)


class TestMape(unittest.TestCase):

    def test_random_lists(self):
        predictions = [1.2, 3.4, 7.1, 0.9]
        true_values = [0.4, 5.6, 4.5, 8.7]
        result = mape(predictions, true_values)
        expected_result = (80/0.4 + 220/5.6 + 260/4.5 + 780/8.7) / 4
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_l1_norm(self):
        predictions = [1.2, 3.4, -7.1, 0.9, 5.7, -0.76]
        true_values = [0]*6
        result = mape(predictions, true_values)
        expected_result = (120 + 340 + 710 + 90 + 570 + 76)*1e8 / 6
        self.assertAlmostEqual(result, expected_result, places=5)

    def test_perfect_accuracy(self):
        true_values = [0.4, 5.6, 4.5, 8.7]
        result = mape(true_values, true_values)
        expected_result = 0
        self.assertEqual(result, expected_result)

    def test_zero_length(self):
        predictions = []
        true_values = []
        test = False
        try:
            result = mape(predictions, true_values)
        except ZeroDivisionError:
            test = True
        self.assertTrue(test)


class TestMse(unittest.TestCase):

    def test_random_lists(self):
        predictions = [1.2, 3.4, 7.1, 0.9]
        true_values = [0.4, 5.6, 4.5, 8.7]
        result = mse(predictions, true_values)
        expected_result = (0.8**2 + 2.2**2 + 2.6**2 + 7.8**2) / 4
        self.assertAlmostEqual(result, expected_result)

    def test_l2_norm(self):
        predictions = [1.2, 3.4, -7.1, 0.9, 5.7, -0.76]
        true_values = [0]*6
        result = mse(predictions, true_values)
        expected_result = (1.2**2 + 3.4**2 + 7.1**2 + 0.9**2 + 5.7**2 + 0.76**2) / 6
        self.assertAlmostEqual(result, expected_result)

    def test_perfect_accuracy(self):
        true_values = [0.4, 5.6, 4.5, 8.7]
        result = mse(true_values, true_values)
        expected_result = 0
        self.assertAlmostEqual(result, expected_result)

    def test_zero_length(self):
        predictions = []
        true_values = []
        test = False
        try:
            result = mse(predictions, true_values)
        except ZeroDivisionError:
            test = True
        self.assertTrue(test)


class TestMsle(unittest.TestCase):

    def test_random_lists(self):
        predictions = [1.2, 3.4, 7.1, 0.9]
        true_values = [0.4, 5.6, 4.5, 8.7]
        result = msle(predictions, true_values)
        expected_result = (log(1.4/2.2)**2 + log(6.6/4.4)**2 + log(5.5/8.1)**2 + log(9.7/1.9)**2) / 4
        self.assertAlmostEqual(result, expected_result)

    def test_l2_norm(self):
        predictions = [1.2, 3.4, 7.1, 0.9, 5.7, 0.76]
        true_values = [0]*6
        result = msle(predictions, true_values)
        expected_result = (log(2.2)**2 + log(4.4)**2 + log(8.1)**2 + log(1.9)**2 + log(6.7)**2 + log(1.76)**2) / 6
        self.assertAlmostEqual(result, expected_result)

    def test_perfect_accuracy(self):
        true_values = [0.4, 5.6, 4.5, 8.7]
        result = msle(true_values, true_values)
        expected_result = 0
        self.assertAlmostEqual(result, expected_result)

    def test_zero_length(self):
        predictions = []
        true_values = []
        test = False
        try:
            result = msle(predictions, true_values)
        except ZeroDivisionError:
            test = True
        self.assertTrue(test)

    def test_negative_values(self):
        predictions = [-3]
        true_values = [-4]
        test = False
        try:
            result = msle(predictions, true_values)
        except ValueError as inst:
            self.assertEqual(inst.args[0], 'math domain error')
            test = True
        self.assertTrue(test)


class TestLogCosh(unittest.TestCase):

    def test_random_lists(self):
        predictions = [1.2, 3.4, 7.1, 0.9]
        true_values = [0.4, 5.6, 4.5, 8.7]
        result = logcosh(predictions, true_values)
        expected_result = (log(cosh(0.8)) + log(cosh(2.2)) + log(cosh(2.6)) + log(cosh(7.8))) / 4
        self.assertAlmostEqual(result, expected_result)

    def test_l2_norm(self):
        predictions = [1.2, 3.4, 7.1, 0.9, 5.7, 0.76]
        true_values = [0]*6
        result = logcosh(predictions, true_values)
        expected_result = (log(cosh(1.2)) + log(cosh(3.4)) + log(cosh(7.1)) + log(cosh(0.9)) + log(cosh(5.7)) + log(cosh(0.76))) / 6
        self.assertAlmostEqual(result, expected_result)

    def test_perfect_accuracy(self):
        true_values = [0.4, 5.6, 4.5, 8.7]
        result = logcosh(true_values, true_values)
        expected_result = 0
        self.assertAlmostEqual(result, expected_result)

    def test_zero_length(self):
        predictions = []
        true_values = []
        test = False
        try:
            result = logcosh(predictions, true_values)
        except ZeroDivisionError:
            test = True
        self.assertTrue(test)


DEFAULT_DELTA = 1.35
DEFAULT_DELTA_TERM = 0.5*(1.35**2)
HIGHER_DELTA = 2.4
HIGHER_DELTA_TERM = 0.5*(HIGHER_DELTA**2)
LOWER_DELTA = 0.1
LOWER_DELTA_TERM = 0.5*(LOWER_DELTA**2)
class TestHuber(unittest.TestCase):

    def test_random_lists_default_delta(self):
        predictions = [1.2, 3.4, 7.1, 0.9]
        true_values = [0.4, 5.6, 4.5, 8.7]
        result = huber(predictions, true_values)
        error1 = 0.5*(0.8**2)
        error2 = DEFAULT_DELTA*2.2 - DEFAULT_DELTA_TERM
        error3 = DEFAULT_DELTA*2.6 - DEFAULT_DELTA_TERM
        error4 = DEFAULT_DELTA*7.8 - DEFAULT_DELTA_TERM
        expected_result = (error1 + error2 + error3 + error4) / 4
        self.assertAlmostEqual(result, expected_result)

    def test_random_lists_higher_delta(self):
        predictions = [1.2, 3.4, 7.1, 0.9]
        true_values = [0.4, 5.6, 4.5, 8.7]
        result = huber(predictions, true_values, delta=HIGHER_DELTA)
        error1 = 0.5*(0.8**2)
        error2 = 0.5*(2.2**2)
        error3 = HIGHER_DELTA*2.6 - HIGHER_DELTA_TERM
        error4 = HIGHER_DELTA*7.8 - HIGHER_DELTA_TERM
        expected_result = (error1 + error2 + error3 + error4) / 4
        self.assertAlmostEqual(result, expected_result)

    def test_l2_norm_default_delta(self):
        predictions = [1.2, 3.4, 7.1, 0.9, 5.7, 0.76]
        true_values = [0]*6
        result = huber(predictions, true_values)
        error1 = 0.5*(1.2**2)
        error2 = DEFAULT_DELTA*3.4 - DEFAULT_DELTA_TERM
        error3 = DEFAULT_DELTA*7.1 - DEFAULT_DELTA_TERM
        error4 = 0.5*(0.9**2)
        error5 = DEFAULT_DELTA*5.7 - DEFAULT_DELTA_TERM
        error6 = 0.5*(0.76**2)
        expected_result = (error1 + error2 + error3 + error4 + error5 + error6) / 6
        self.assertAlmostEqual(result, expected_result)

    def test_l2_norm_lower_delta(self):
        predictions = [1.2, 3.4, 7.1, 0.9, 5.7, 0.76]
        true_values = [0]*6
        result = huber(predictions, true_values, delta=0.1)
        error1 = LOWER_DELTA*1.2 - LOWER_DELTA_TERM
        error2 = LOWER_DELTA*3.4 - LOWER_DELTA_TERM
        error3 = LOWER_DELTA*7.1 - LOWER_DELTA_TERM
        error4 = LOWER_DELTA*0.9 - LOWER_DELTA_TERM
        error5 = LOWER_DELTA*5.7 - LOWER_DELTA_TERM
        error6 = LOWER_DELTA*0.76 - LOWER_DELTA_TERM
        expected_result = (error1 + error2 + error3 + error4 + error5 + error6) / 6
        self.assertAlmostEqual(result, expected_result)

    def test_perfect_accuracy(self):
        true_values = [0.4, 5.6, 4.5, 8.7]
        result = huber(true_values, true_values)
        expected_result = 0
        self.assertAlmostEqual(result, expected_result)

    def test_zero_length(self):
        predictions = []
        true_values = []
        test = False
        try:
            result = huber(predictions, true_values)
        except ZeroDivisionError:
            test = True
        self.assertTrue(test)


class TestPoisson(unittest.TestCase):

    def test_random_lists(self):
        predictions = [1.2, 3.4, 7.1, 0.9]
        true_values = [0.4, 5.6, 4.5, 8.7]
        result = poisson(predictions, true_values)
        error1 = 1.2 - 0.4*log(1.2)
        error2 = 3.4 - 5.6*log(3.4)
        error3 = 7.1 - 4.5*log(7.1)
        error4 = 0.9 - 8.7*log(0.9)
        expected_result = (error1 + error2 + error3 + error4) / 4
        self.assertAlmostEqual(result, expected_result)

    def test_l2_norm(self):
        predictions = [1.2, 3.4, 7.1, 0.9, 5.7, 0.76]
        true_values = [0]*6
        result = poisson(predictions, true_values)
        expected_result = sum(predictions) / 6
        self.assertAlmostEqual(result, expected_result)

    def test_perfect_accuracy(self):
        true_values = [0.4, 5.6, 4.5, 8.7]
        result = poisson(true_values, true_values)
        error1 = 0.4 - 0.4*log(0.4)
        error2 = 5.6 - 5.6*log(5.6)
        error3 = 4.5 - 4.5*log(4.5)
        error4 = 8.7 - 8.7*log(8.7)
        expected_result = (error1 + error2 + error3 + error4)/4
        self.assertAlmostEqual(result, expected_result)

    def test_zero_length(self):
        predictions = []
        true_values = []
        test = False
        try:
            result = poisson(predictions, true_values)
        except ZeroDivisionError:
            test = True
        self.assertTrue(test)

if __name__ == '__main__':
    # Run tests
    unittest.main()
