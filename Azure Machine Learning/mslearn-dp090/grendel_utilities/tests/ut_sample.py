import unittest
import os
import numpy as np

class TestStringMethods(unittest.TestCase):
    def setUp(self):
        self.currdir = os.getcwd () 

    @unittest.skip("skipping...")
    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    @unittest.skip("skipping...")
    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    @unittest.skip("skipping...")
    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def test_concatenate_arrays(self):
        xf1 = [[1,2,3,4,5,6,7],[11,12,13,14,15,16,17]]
        xf2 = [[51,52,53,54,55,56],[61,62,63,64,65,66]]
        xf = np.concatenate((xf1,xf2),axis=1)
        print(xf)
        self.asserTrue(3==3)

if __name__ == '__main__':
    unittest.main()
