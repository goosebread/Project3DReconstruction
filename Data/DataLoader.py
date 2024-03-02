#example
import cv2 
import sys
import os 

datasetsPath = os.path.join(sys.path[0], 'Datasets')

imgPath = os.path.join(datasetsPath,"StatueOfLiberty","IM_69.jpg")

# Reading an image
img = cv2.imread(imgPath) 
image_scale_down = 10
x = (int)(img.shape[0]/image_scale_down)
y = (int)(img.shape[1]/image_scale_down)
image = cv2.resize(img, (x,y))

  
# Using cv2.imshow() method 
# Displaying the image 
window_name = 'image'
cv2.imshow(window_name, image) 
  
# waits for user to press any key 
# (this is necessary to avoid Python kernel form crashing) 
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows() 