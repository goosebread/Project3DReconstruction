#example
import cv2 

path = R"C:\Code\3DReconstruction\Data\Datasets\StatueOfLiberty\IM_69.jpg"

# Reading an image in default mode 
img = cv2.imread(path) 
image_scale_down = 10
x = (int)(img.shape[0]/image_scale_down)
y = (int)(img.shape[1]/image_scale_down)
image = cv2.resize(img, (x,y))

# Window name in which image is displayed 
window_name = 'image'
  
# Using cv2.imshow() method 
# Displaying the image 
cv2.imshow(window_name, image) 
  
# waits for user to press any key 
# (this is necessary to avoid Python kernel form crashing) 
cv2.waitKey(0) 
  
# closing all open windows 
cv2.destroyAllWindows() 