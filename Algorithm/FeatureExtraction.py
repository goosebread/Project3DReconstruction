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
    pass

#https://ieeexplore.ieee.org/document/5204091
def opponentSIFT(img):
    #Step 1 convert to opponent color space
    #TODO optimize
    B = img[:,:,0]
    G = img[:,:,1]
    R = img[:,:,2]

    O1 = np.divide((R-G),np.sqrt(2))
    O2 = np.divide((R+G-2*B),np.sqrt(6))
    O3 = np.divide((R+G+B),np.sqrt(3))
    cv.imwrite('sift_keypointsO1.jpg',np.uint8(O1))
    cv.imwrite('sift_keypointsO2.jpg',np.uint8(O2))
    cv.imwrite('sift_keypointsO3.jpg',np.uint8(O3))

    #Step 2 use Harris-Laplace point detector on intensity channel (o3)
    #TODO use a real point detector or figure out what parameters to use with cv SIFT
    #use this space to specify additional parameters
    sift = cv.SIFT_create(nfeatures=1000)
    #sift = cv.SIFT_create(nfeatures=1000, nOctaveLayers=3, sigma=10)

    kp = sift.detect(np.uint8(O3),None)

    #Step 3 compute descriptors for each opponent channel
    _,des1 = sift.compute(np.uint8(O1),kp)
    _,des2 = sift.compute(np.uint8(O2),kp)
    _,des3 = sift.compute(np.uint8(O3),kp)

    #combine into one large descriptor
    des = np.concatenate((des1,des2,des3),axis=1)

    return kp, des
