from cross_sampling import cross_sample_tensor
import unittest
import numpy as np


class Test_CrossSampling(unittest.TestCase):

    def test_3_by_3(self):
        #Define tensor
        tensor = np.array([i for i in range(64)]).reshape((4,4,4))
        
        #Try with default rank
        b, j, a, n = cross_sample_tensor(tensor)
        self.assertEqual(b.shape, (1,1,1))
        for arm in a:
            self.assertEqual(arm.shape, (4,1))
        for joint in j:
            self.assertEqual(joint.shape, (1,1))
        self.assertEqual(n, 10)

        #Try with custom rank
        b, j, a, n = cross_sample_tensor(tensor, tucker_rank_list=[4,2,3])
        self.assertEqual(b.shape, (4,2,3))
        flag_1, flag_2, flag_3 = False, False, False
        for arm in a:
            if arm.shape == (4,4):
                flag_1 = True
            if arm.shape == (4,2):
                flag_2 = True
            if arm.shape == (4,3):
                flag_3 = True
        self.assertTrue(flag_1)
        self.assertTrue(flag_2) 
        self.assertTrue(flag_3)

        flag_1, flag_2, flag_3 = False, False, False
        for joint in j:
            if joint.shape == (4,4):
                flag_1 = True
            if joint.shape == (2,2):
                flag_2 = True
            if joint.shape == (3,3):
                flag_3 = True
        self.assertTrue(flag_1)
        self.assertTrue(flag_2) 
        self.assertTrue(flag_3)

        self.assertEqual(n, (4*2*3 + 4*(4-4) + 2*(4-2) + 3*(4-3)))


        #Try another custom rank
        b, j, a, n = cross_sample_tensor(tensor, tucker_rank_list=[3,1,2])
        self.assertEqual(b.shape, (3,1,2))
        flag_1, flag_2, flag_3 = False, False, False
        for arm in a:
            if arm.shape == (4,2):
                flag_1 = True
            if arm.shape == (4,1):
                flag_2 = True
            if arm.shape == (4,2):
                flag_3 = True
        self.assertTrue(flag_1)
        self.assertTrue(flag_2) 
        self.assertTrue(flag_3)

        flag_1, flag_2, flag_3 = False, False, False
        for joint in j:
            if joint.shape == (3,2):
                flag_1 = True
            if joint.shape == (1,1):
                flag_2 = True
            if joint.shape == (2,2):
                flag_3 = True
        self.assertTrue(flag_1)
        self.assertTrue(flag_2) 
        self.assertTrue(flag_3)

        self.assertEqual(n, (3*1*2 + 2*(4-3) + 1*(4-1) + 2*(4-2)))

    def test_4_by_4(self):
        #Define tensor
        tensor = np.array([i for i in range(64)]).reshape((2,4,4,2))
        
        #Try with default rank
        b, j, a, n = cross_sample_tensor(tensor)
        self.assertEqual(b.shape, (1,1,1,1))
        count_1 = 0
        count_2 = 0
        for arm in a:
            if arm.shape == (2,1):
                count_1 += 1
            if arm.shape == (4,1):
                count_2 += 1
        self.assertEqual(count_1, 2)
        self.assertEqual(count_2, 2)
        for joint in j:
            self.assertEqual(joint.shape, (1,1))
        self.assertEqual(n, 1 + 1 + 3 + 1 + 3)

        #Try with custom rank
        b, j, a, n = cross_sample_tensor(tensor, tucker_rank_list=[2,4,3,2])
        self.assertEqual(b.shape, (2,4,3,2))
        flag_1, flag_2, flag_3, flag_4 = False, False, False, False
        for arm in a:
            if arm.shape == (2,2):
                flag_1 = True
            if arm.shape == (4,4):
                flag_2 = True
            if arm.shape == (4,3):
                flag_3 = True
            if arm.shape == (2,2):
                flag_4 = True
        self.assertTrue(flag_1)
        self.assertTrue(flag_2) 
        self.assertTrue(flag_3)
        self.assertTrue(flag_4)

        flag_1, count_2, flag_3 = False, 0, False
        for joint in j:
            if joint.shape == (4,4):
                flag_1 = True
            if joint.shape == (2,2):
                count_2 += 1
            if joint.shape == (3,3):
                flag_3 = True
        self.assertTrue(flag_1)
        self.assertEqual(count_2, 2) 
        self.assertTrue(flag_3)

        self.assertEqual(n, (2*4*3*2 + 2*(2-2) + 4*(4-4) + 3*(4-3) + 2*(2-2)))


        #Try another custom rank
        b, j, a, n = cross_sample_tensor(tensor, tucker_rank_list=[1,1,4,2])
        self.assertEqual(b.shape, (1,1,4,2))
        flag_1, flag_2, flag_3, flag_4 = False, False, False, False
        for arm in a:
            if arm.shape == (2,1):
                flag_1 = True
            if arm.shape == (4,1):
                flag_2 = True
            if arm.shape == (4,2):
                flag_3 = True
            if arm.shape == (2,2):
                flag_4 = True
        self.assertTrue(flag_1)
        self.assertTrue(flag_2) 
        self.assertTrue(flag_3)
        self.assertTrue(flag_4)

        flag_1, flag_2, flag_3, flag_4 = False, False, False, False
        for joint in j:
            if joint.shape == (1,1):
                flag_1 = True
            if joint.shape == (1,1):
                flag_2 = True
            if joint.shape == (2,2):
                flag_3 = True
            if joint.shape == (4,2):
                flag_4 = True
        self.assertTrue(flag_1)
        self.assertTrue(flag_2) 
        self.assertTrue(flag_3)
        self.assertTrue(flag_4)

        self.assertEqual(n, (1*1*4*2 + 1*(2-1) + 1*(4-1) + 2*(4-4) + 2*(2-2)))

if __name__ == '__main__':
    unittest.main()
