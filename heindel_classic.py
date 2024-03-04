import cv2
import os
import numpy as np
import tifffile as tf

def increase_contrast(image, clip_limit=6.0, grid_size=(10, 10)):
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
    contrast_image = increase_contrast(image)
    contrast_image = cv2.GaussianBlur(contrast_image, (11, 11), 0)

#     grad_x = cv2.Sobel(contrast_blur, cv2.CV_32F, 1, 0, ksize=3, scale=1)
#     grad_y = cv2.Sobel(contrast_blur, cv2.CV_32F, 0, 1, ksize=3, scale=1)

#     # Combine gradient images
#     gradient_image = cv2.magnitude(grad_x, grad_y)
#     gradient_image = np.uint8(255 * gradient_image / np.max(gradient_image))

# # Display the original and edges-detected images
#     cv2.imshow('Edges Detected (Sobel)', gradient_image)
    
    cv2.imshow('og image', image)
    cv2.imshow('contrast', contrast_image)
    
    # Apply Gaussian blur to reduce noise
    
    # Threshold the image to create a binary image
    binary_image = cv2.adaptiveThreshold(contrast_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 51, 2)
    binary_image = cv2.dilate(cv2.erode(binary_image, np.ones((7,7))),np.ones((7,7)))

    num_ccs, ccs, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
    # print(max(stats[1:,cv2.CC_STAT_AREA]))
    # print(min(stats[1:,cv2.CC_STAT_AREA]))
    # print(np.median(stats[1:,cv2.CC_STAT_AREA]))
    # print(np.quantile(stats[:,cv2.CC_STAT_AREA], q= 0.999))

    blank = np.ones_like(image)*255
    for i in range(1, num_ccs):
        if stats[i,cv2.CC_STAT_AREA] > 250:
            left = stats[i,cv2.CC_STAT_LEFT]
            top = stats[i,cv2.CC_STAT_TOP]
            height = stats[i,cv2.CC_STAT_HEIGHT]
            width = stats[i,cv2.CC_STAT_WIDTH]
            blank[top:top+height, left:left+width] = contrast_image[top:top+height, left:left+width]

    cv2.imshow("lol", blank)

    cv2.imshow('wind', binary_image)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image,contours)
    droplets = []
    for contour in contours:
        # Calculate the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w
        
        # Check if aspect ratio is within the specified range
        if aspect_ratio >= aspect_ratio_range[0] and aspect_ratio <= aspect_ratio_range[1]:
            droplets.append((x, y, w, h))
            cv2.drawContours(contrast_image, [contour], -1, 0, 1)
    cv2.imshow('cont', contrast_image)

    return droplets

# Example usage:
image_path = 'testdata\\512_608_x=-2.08_y=31.2_Ql=0.099_Qg=75_Qsw=75_fps=6000_000141.tif'
droplets = identify_droplets(image_path)

print("Number of droplets identified:", len(droplets))
print("Coordinates of droplets (x, y, width, height):")
# for i, (x, y, w, h) in enumerate(droplets):
#     print(f"Droplet {i+1}: ({x}, {y}, {w}, {h})")

cv2.waitKey(0)
cv2.destroyAllWindows()
# def find_round_objects(image):
#     # Convert image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
#     # Detect circles using HoughCircles
#     circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
#                                param1=25, param2=25, minRadius=2, maxRadius=250)
    
#     # If circles found, draw bounding boxes and count
#     if circles is not None:
#         circles = circles[0]
#         num_circles = len(circles)
#         circle_info = []
        
#         for circle in circles:
#             #  Convert circle parameters to integers
#             x, y, r = map(int, circle)
#             # Draw circle
#             # Calculate bounding box coordinates
#             x1, y1 = int(x - r), int(y - r)
#             x2, y2 = int(x + r), int(y + r)
#             # Draw bounding box
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
#         return num_circles, (x1, y1), (x2, y2)
            
#     else:
#         return 0, None

# def process_images_in_directory(directory):
#     for filename in os.listdir(directory):
#         if filename.endswith('.tif'):
#             filepath = os.path.join(directory, filename)
#             image = cv2.imread(filepath)
#             if image is not None:
#                 num_circles, circle_info, _ = find_round_objects(image)
#                 if num_circles > 0:
#                     # Add metadata to TIFF file
#                     cv2.imshow('wind', image)
#                     cv2.waitKey(0)
#                     cv2.destroyAllWindows()
#                 else:
#                     print(f"No circles found in {filename}")
#             else:
#                 print(f"Unable to read image: {filename}")

# # Directory containing the .tif files
# data_directory = 'testdata'

# # Process images in the directory
# process_images_in_directory(data_directory)

# import cv2
# import os

# def find_round_objects(image):
#     # Convert image to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Apply Gaussian blur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    
#     # Detect circles using HoughCircles
#     circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
#                                param1=25, param2=15, minRadius=2, maxRadius=500)
    
#     # If circles found, draw bounding boxes and count
#     if circles is not None:
#         circles = circles[0]
#         num_circles = len(circles)
        
#         for circle in circles:
#             # Convert circle parameters to integers
#             x, y, r = map(int, circle)
#             # Draw circle
#             cv2.circle(image, (x, y), r, (0, 255, 0), 4)
#             # Calculate bounding box coordinates
#             x1, y1 = int(x - r), int(y - r)
#             x2, y2 = int(x + r), int(y + r)
#             # Draw bounding box
#             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
#         return num_circles, (x1, y1), (x2, y2)
#     else:
#         return 0, None, None

# def process_images_in_directory(directory):
#     for filename in os.listdir(directory):
#         if filename.endswith('.tif'):
#             filepath = os.path.join(directory, filename)
#             print(filepath)
#             image = cv2.imread(filepath)
#             if image is not None:
#                 num_circles, upper_left, lower_right = find_round_objects(image)
#                 print(f"Image: {filename}, Circles found: {num_circles}, Bounding Box: {upper_left} - {lower_right}")
#                 # Optionally, you can save or overwrite the modified image here
#                 cv2.imshow(image)
#                 cv2.waitKey(0)
#                 cv2.destroyAllWindows()
#             else:
#                 print(f"Unable to read image: {filename}")

# # Directory containing the .tif files
# data_directory = 'testdata'

# # Process images in the directory
# process_images_in_directory(data_directory)
