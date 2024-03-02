import sys
import unittest
sys.path.append(R"C:\Code\3DReconstruction")

from Algorithm.Model import exampleFunc

class ExampleUnittest(unittest.TestCase):
    def testExample(self):
        self.assertEqual(exampleFunc(1,2), 3)

if __name__ == '__main__':
    unittest.main()