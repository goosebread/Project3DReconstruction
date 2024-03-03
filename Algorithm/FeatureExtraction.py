# algorithm here takes in 2D images and returns some feature vector
import numpy as np
import cv2 as cv

#Original SIFT Paper
#https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
#TODO if SIFT proves to be a bottleneck for the system, we can try to compute keypoints in parallel

#This function is a wrapper for the OpenCV SIFT implementation
#If we need to implement SIFT ourselves, this could be useful as something to compare against
#Input: cv2 image
#return: 
#kp - a list of keypoints 
#des - a numpy array of shape (Number of Keypoints)×128.
def referenceSIFT(img):
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #use this space to specify additional parameters

    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray,None)
    return kp, des


#k=2 KNN with ratio test
#We can probably use parallelism to conpute this faster
#inputs: two sets of keypoints and descriptors
#return: set of matches
def featureMatcher(kp1, des1, kp2, des2):
    #for each kp1:
        #for each kp2:
            #calc distance/l2 norm
            #track best 2 matches
        #calculate ratio and compare against threshold
        #tau = 0.8 for euclidean distance, 0.64 for L2 norm
        #if ratio test passed, add to list or dictionary
    #return list or dictionary