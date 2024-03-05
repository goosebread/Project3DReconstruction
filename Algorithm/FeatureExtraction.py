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

    sift = cv.SIFT_create(nfeatures=1000)
    kp, des = sift.detectAndCompute(gray,None)
    return kp, des


#k=2 KNN with ratio test
#We can probably use parallelism to conpute this faster
#inputs: two sets of keypoints and descriptors
#return: set of matches
def featureMatcher(kp1, des1, kp2, des2, threshold=0.8):
    matches = []
    print(len(kp1))
    for i_index in range(len(kp1)):
        print(i_index)
        best=np.inf
        bestJIndex=-1
        secondBest=np.inf
        for j_index in range(len(kp2)):
            #calc l2 norm
            distanceSquared = np.linalg.norm(des1[i_index] - des2[j_index], ord=2)
            #track best 2 matches
            if distanceSquared<=best:
                bestJIndex=j_index
                secondBest=best
                best=distanceSquared
            elif distanceSquared<=secondBest:
                secondBest=distanceSquared
        #calculate ratio and compare against threshold
        ratio = best/secondBest
        #tau = 0.8 for euclidean distance, 0.64 for L2 norm
        #if ratio test passed, add to list or dictionary
        if ratio<threshold**2:
            matches.append((kp1[i_index], kp2[bestJIndex]))
    #return list or dictionary
    return matches

def outlierRemover(matches):
    #use RANSAC
    #use 8 point correspondances to estimate a fundamental matrix
    #need some sort of match score for each fundamental matrix
    