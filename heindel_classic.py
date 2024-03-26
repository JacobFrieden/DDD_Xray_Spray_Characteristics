# TODO: Can we contrast our way into oblivion with correct grid sizes to actually get just targets
# The fact only 6 people showed up is incredible
# Mayhaps a mask R-CNN finetuning is actually relevant.
import cv2
import os
import numpy as np
import tifffile as tf

def increase_contrast(image, clip_limit=8.0, grid_size=(15, 15)):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    equalized = clahe.apply(gray)
    
    # If the original image was in color, convert the equalized image back to color
    if len(image.shape) == 3:
        equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    
    return equalized



def identify_droplets(image_path, aspect_ratio_range=(0.8, 1.2)):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    imageb = cv2.GaussianBlur(image, (9,9), 0)
    contrast_image = increase_contrast(imageb, grid_size=(29,29))
    contrast_image2 = increase_contrast(contrast_image, grid_size=(5,5))
    # cv2.imshow('contrast2', contrast_image2)
    contrast_image2b = cv2.GaussianBlur(contrast_image2, (9,9), 0)
    # cv2.imshow('contrast2b', contrast_image2b)
    _, c2ibw = cv2.threshold(contrast_image2, 86,255,cv2.THRESH_BINARY_INV)
    c2ibw = cv2.erode(c2ibw, np.ones((1,7)))
    c2ibw = cv2.erode(c2ibw, np.ones((7,1)))
    c2ibw = cv2.dilate(c2ibw, np.ones((7,7)))

    # cv2.imshow('normal binary', c2ibw)
    
    # cv2.imshow('og image', image)
    # cv2.imshow('contrast', contrast_image)

    num_ccs, ccs, stats, centroids = cv2.connectedComponentsWithStats(c2ibw)
    cv2.imwrite(f'training_data/masks/{image_path[-10:-4]}.png', ccs)
    # print(max(stats[1:,cv2.CC_STAT_AREA]))
    # print(min(stats[1:,cv2.CC_STAT_AREA]))
    # print(np.median(stats[1:,cv2.CC_STAT_AREA]))
    # print(np.quantile(stats[:,cv2.CC_STAT_AREA], q= 0.999))

    # blank = np.zeros_like(image)
    # blank[c2ibw==255] = contrast_image[c2ibw==255]


    # cv2.imshow("lol", blank)
    # cv2.waitKey(0)

    # contrast_colorized = cv2.cvtColor(contrast_image, cv2.COLOR_GRAY2BGR)
    # # Find contours in the binary image
    # contours, _ = cv2.findContours(c2ibw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # # cv2.drawContours(image,contours)
    # droplets = []
    # for contour in contours:
    #     # Calculate the bounding box of the contour
    #     x, y, w, h = cv2.boundingRect(contour)
    #     aspect_ratio = h / w
        
    #     # Check if aspect ratio is within the specified range
    #     if aspect_ratio >= aspect_ratio_range[0] and aspect_ratio <= aspect_ratio_range[1] and w*h > 40:
    #         droplets.append((x, y, w, h))
    #         cv2.drawContours(contrast_colorized, [contour], -1, [0,0,255], 1)

    # cv2.imshow('cont', contrast_colorized)
    # cv2.imwrite('Contours.png', contrast_colorized)
    # return droplets

# Example usage:
IMAGERY_DIR = 'training_data/IMGs'
for file in os.listdir(os.path.join('',IMAGERY_DIR)):
    identify_droplets(os.path.join(IMAGERY_DIR, file))
