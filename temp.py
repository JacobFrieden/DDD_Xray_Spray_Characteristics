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
    image = cv2.GaussianBlur(image, (7,7), 0)
    contrast_image = increase_contrast(image, grid_size=(35,35))
    cv2.imshow('c1', contrast_image)
    contrast_image2 = increase_contrast(contrast_image, grid_size=(5,5))
    cv2.imshow('c2', contrast_image2)



# Example usage:
image_path = 'testdata\\512_608_x=-2.08_y=31.2_Ql=0.099_Qg=75_Qsw=75_fps=6000_000141.tif'
identify_droplets(image_path, (0.5, 1.5))
cv2.waitKey(0)
cv2.destroyAllWindows()