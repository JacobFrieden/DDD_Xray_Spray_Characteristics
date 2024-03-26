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



def generate_masks(image_path, output_dir = 'training_data\\masks'):
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
    c2ibw = cv2.dilate(c2ibw, np.ones((11,11)))
    # cv2.imshow('bw', c2ibw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    num_ccs, ccs, stats, centroids = cv2.connectedComponentsWithStats(c2ibw)
    output_loc = os.path.join(output_dir,image_path.split(os.sep)[-1][:-4],'_mask.png')
    print(f"Outputting mask to {os.path.join(output_dir,image_path.split(os.sep)[-1][-10:-4]+'_mask.png')}...", end=' ')
    flag = cv2.imwrite(output_loc, ccs)
    print(flag)


