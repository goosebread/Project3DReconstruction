import sys
import os 
import unittest
import cv2 as cv

#gotta mess around with the path structure a bit to expose things for unit tests
projectPath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(projectPath)

from Algorithm.FeatureExtraction import referenceSIFT

class TestSIFT(unittest.TestCase):
    def testImage1(self):
        imgPath = os.path.join(projectPath,'Algorithm','tests','test1.jpg')
        img = cv.imread(imgPath)
        kp, des = referenceSIFT(img)
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        #img2=cv.drawKeypoints(gray,kp,img)
        img2=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite('sift_keypoints.jpg',img2)

        cv.imshow('Display SIFT Output', img2) 
        cv.waitKey(0) 
        cv.destroyAllWindows() 

    def testStar(self):
        imgPath = os.path.join(projectPath,'Algorithm','tests','star.png')
        img = cv.imread(imgPath)
        kp, des  = referenceSIFT(img)
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        #img2=cv.drawKeypoints(gray,kp,img)
        img2=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite('sift_keypoints.jpg',img2)

        cv.imshow('Display SIFT Output', img2) 
        cv.waitKey(0) 
        cv.destroyAllWindows() 
        self.assertEqual(3, 3)
if __name__ == '__main__':
    unittest.main()