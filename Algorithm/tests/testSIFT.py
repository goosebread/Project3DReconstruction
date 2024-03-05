import sys
import os 
import unittest
import cv2 as cv

#gotta mess around with the path structure a bit to expose things for unit tests
projectPath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(projectPath)

from Algorithm.FeatureExtraction import *

class TestSIFT(unittest.TestCase):
    def testImage1(self):
        imgPath = os.path.join(projectPath,'Algorithm','tests','test1.jpg')
        img = cv.imread(imgPath)
        kp, des = referenceSIFT(img)
        gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

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

        img2=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite('sift_keypoints.jpg',img2)

        cv.imshow('Display SIFT Output', img2) 
        cv.waitKey(0) 
        cv.destroyAllWindows() 
        self.assertEqual(3, 3)

    def testFeatureMatching(self):
        imgPath1 = os.path.join(projectPath,'Algorithm','tests','dino1.jpg')
        imgPath2 = os.path.join(projectPath,'Algorithm','tests','dino2.jpg')

        img1 = cv.imread(imgPath1)
        img2 = cv.imread(imgPath2)

        kp1, des1  = referenceSIFT(img1)
        kp2, des2  = referenceSIFT(img2)

        #assuming same image size for visualization of matched points
        _,widthOffset,_ = img1.shape

        h_img = cv.hconcat([img1, img2])
        matches = featureMatcher(kp1,des1,kp2,des2)

        #expects array of array of points
        matches = list(map(lambda match: np.array([match[0].pt, (match[1].pt[0]+widthOffset, match[1].pt[1])], np.int32), matches))

        #draw matching point lines on concatenated image
        finalImg = cv.polylines(h_img, matches, False, (255,0,0))

        cv.imwrite('sift_keypoints.jpg',finalImg)

        cv.imshow('Display SIFT Output', finalImg) 
        cv.waitKey(0) 
        cv.destroyAllWindows() 
        self.assertEqual(3, 3)
        
if __name__ == '__main__':
    unittest.main()