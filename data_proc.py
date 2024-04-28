"""
    Splits files in DATASOURCE into train and test,
    generates the thresholding based masks, and augments the training
    data using random cropping, flipping, and gaussian noising
"""


import cv2
import os
import numpy as np
import tifffile as tf
import tqdm

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

def rand_crop(img, mask):
    crop_size = np.random.randint(100,401)
    c1 = np.random.randint(0,img.shape[0]-crop_size)
    c2 = np.random.randint(0, img.shape[1]-crop_size)
    img = img[c1:c1+crop_size, c2:c2+crop_size]
    mask = mask[c1:c1+crop_size, c2:c2+crop_size]

    # Cropping can do funky things that result in 0-area bboxes 
    # that MRCNN can't handle, this compensates
    mask = cv2.dilate(cv2.erode(mask, np.ones((3,3))),np.ones((3,3)))
    return img, mask

def rand_flip(img, mask):
    flip_code = np.random.choice([-1,0,1])
    return cv2.flip(img, flip_code), cv2.flip(mask, flip_code)

def generate_binary_mask(image):
    imageb = cv2.GaussianBlur(image, (9,9), 0)
    contrast_image = increase_contrast(imageb, grid_size=(29,29))
    contrast_image2 = increase_contrast(contrast_image, grid_size=(5,5))
    # cv2.imshow('contrast2', contrast_image2)
    # cv2.imshow('contrast2b', contrast_image2b)
    _, c2ibw = cv2.threshold(contrast_image2, 86,255,cv2.THRESH_BINARY_INV)
    c2ibw = cv2.erode(c2ibw, np.ones((1,7)))
    c2ibw = cv2.erode(c2ibw, np.ones((7,1)))
    c2ibw = cv2.dilate(c2ibw, np.ones((7,7)))
    return c2ibw

if __name__ == "__main__": 

    DATASOURCE = 'Data/'
    TRAINING_DIR = 'Processed Data/CE_PREPROC/train/'
    TESTING_DIR = 'Processed Data/CE_PREPROC/test/'
    # os.makedirs(os.path.join(TRAINING_DIR, 'IMGs'), exist_ok=True)
    # os.makedirs(os.path.join(TRAINING_DIR, 'masks'), exist_ok=True)
    # os.makedirs(os.path.join(TESTING_DIR, 'IMGs'), exist_ok=True)
    # os.makedirs(os.path.join(TESTING_DIR, 'masks'), exist_ok=True)

    if len(os.listdir(TRAINING_DIR+'IMGs')) + len(os.listdir(TESTING_DIR+'IMGs')) == 0:
        for file in tqdm.tqdm(os.listdir(DATASOURCE)):
            split = np.random.random()
            img = cv2.imread(os.path.join(DATASOURCE, file), cv2.IMREAD_GRAYSCALE)
            img = img[20:-20,:] # Images contain significant noise near top and bottom that couldn't be removed classically
            binary_mask = generate_binary_mask(img)
            if split < 0.8:
                # Training Data
                img = increase_contrast(img, grid_size=(15,15))
                _, cc_mask = cv2.connectedComponents(binary_mask)
                cv2.imwrite(os.path.join(TRAINING_DIR,'IMGs',file[-10:-4]+'.png'), img)
                cv2.imwrite(os.path.join(TRAINING_DIR,'masks',file[-10:-4]+'.png'), cc_mask)
                i = 1
                while i < 5:
                    aug = False
                    aug_img, aug_mask = img, binary_mask
                    if np.random.random() > 0.5:
                        aug = True
                        aug_img, aug_mask = rand_crop(aug_img, aug_mask)
                    if np.random.random() > 0.5:
                        aug = True
                        aug_img, aug_mask = rand_flip(aug_img, aug_mask)
                    if aug:
                        _, aug_mask = cv2.connectedComponents(aug_mask)
                        cv2.imwrite(os.path.join(TRAINING_DIR,'IMGs',f'{file[-10:-4]}_{i}.png'), aug_img)
                        cv2.imwrite(os.path.join(TRAINING_DIR,'masks',f'{file[-10:-4]}_{i}.png'), aug_mask)
                        i+=1
            else:
                # Testing Data
                _, cc_mask = cv2.connectedComponents(binary_mask)
                img = increase_contrast(img, grid_size=(15,15))

                cv2.imwrite(os.path.join(TESTING_DIR,'IMGs',file[-10:-4]+'.png'), img)
                cv2.imwrite(os.path.join(TESTING_DIR,'masks',file[-10:-4]+'.png'), cc_mask)
    else:
        print(f'{len(os.listdir(TRAINING_DIR+"IMGs"))=}')
        print(f'{len(os.listdir(TESTING_DIR+"IMGs"))=}')

