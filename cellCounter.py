# -*- coding: utf-8 -*-
"""
Detect contours, center of mass and surface area of cells from fluoresence microscopy image

@author: cristina
"""

# DEPENDENCIES
import numpy as np
import matplotlib.pyplot as plt
import imutils       # pip install imutils       | url: https://github.com/jrosebr1/imutils
import cv2           # pip install opencv-python | url: https://pypi.python.org/pypi/opencv-python

#%%

# SET UP INPUT DIRECTORY, STORE NAMES OF IMAGES OF INTEREST:

input_dir  = input('Enter the full path for the input folder eg. C:/Users/bruno/Dropbox/ : ')
image_name = input('Enter the name of the image with the filetype suffix eg. figure2b.png :')

# EXAMPLES 
#input_dir       = "C:/Users/bruno/Dropbox/IMAGING_ELECTIVE2017/P2series/"
#image_name      = "Figure2b.PNG"
img_path        = str(input_dir) + str(image_name) 

#READ IMAGE  
img                  = cv2.imread(img_path)
#%%


#1. Convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#2. Apply Gaussian Filter
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#3. Thresholding to binarize
thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]  # min value  = 60, manual inspection of pixels


# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# create a copy of the image to draw contours on
img_with_contours = img.copy()


# loop over the contours and create two maps:
contourCentroid_map = []
contourArea_map     = []

for i in range(0, len(cnts)):
    # compute the centroid of the contour
    M = cv2.moments(cnts[i])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    
    # Store the ith contour's centroid coordinates as a tuple in the list named `contourCentroid_map`
    centroid_coordinates = (cX, cY)
    contourCentroid_map.append(centroid_coordinates)

    # Store the ith contour's area in the list named `contourArea_map`   
    area = cv2.contourArea(cnts[i])
    contourArea_map.append(area)
    
    # draw the contour and center of the shape on the image
    cv2.drawContours(img_with_contours, [cnts[i]], -1, (0, 200, 200), 2)
    cv2.circle(img_with_contours, (cX, cY), 3 , (0, 0, 200), -1)
    cv2.putText(img_with_contours, str(i), (cX - 2, cY - 2), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.45, (200, 200, 50), 1)

# Create a new figure
fig = plt.figure()
fig_size = plt.rcParams["figure.figsize"]


# Plot the images
# Set figure width to 12 and height to 9
fig_size[0] = 20
fig_size[1] = 16
plt.rcParams["figure.figsize"] = fig_size


plt.axis("off")
plt.suptitle('Contours, centroids and index of cells', size = 18)
plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB)) 
plt.savefig( str(input_dir) +'Contours_and_centroids')

#plt.show()


    
#%%

# FUNCTION TO CROP MIN RECTANGLE AROUND IDENTIFIED CONTOUR
def crop_minAreaRect(img, rect):  #returns rotated bounting ractangle around contour

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]

    return img_crop

    
# Call the function crop_minAreaRect while looping over each detectedd contour
original_image =  cv2.imread(img_path)
# Looping over each contour and saving cropped image to input_dir
for i in range(0, len(cnts)):
    
    try:
    
        rect = cv2.minAreaRect(cnts[i])
        
        img_croped = crop_minAreaRect(img_with_contours, rect) ##comment in for output
                
        # Create a new figure
        fig = plt.figure()
        fig_size = plt.rcParams["figure.figsize"]
        
        
        # Plot the images
        # Set figure width to 12 and height to 9
        fig_size[0] = 6
        fig_size[1] = 4
        plt.rcParams["figure.figsize"] = fig_size
        
        plt.title('Cell with index = ' + str(i) + ', centroid = (' + str(contourCentroid_map[i][0]) + ',' + str(int(contourCentroid_map[i][1])) + '), area = ' + str(contourArea_map[i]), size = 14, loc = 'center')
        plt.imshow(cv2.cvtColor (img_croped, cv2.COLOR_BGR2RGB))
        plt.savefig( str(input_dir) +'cell_index_' + str(i) + 'cX_' + str(contourCentroid_map[i][0]) + '_cY_' + str(contourCentroid_map[i][1]) + '_area_' + str(contourArea_map[i]) + '.png', dpi= 400, bbox_inches = 0) # lose the unecessary padding/frame
        plt.show()
    except:
        pass

#%%
        

        