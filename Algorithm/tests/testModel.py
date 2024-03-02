import sys
import os 
import unittest

#gotta mess around with the path structure a bit to expose things for unit tests
projectPath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(projectPath)

from Algorithm.Model import exampleFunc

class ExampleUnittest(unittest.TestCase):
    def testExample(self):
        self.assertEqual(exampleFunc(1,2), 3)

if __name__ == '__main__':
    unittest.main()