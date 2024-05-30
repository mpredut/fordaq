import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_better_mask(image_path, background_path):
    # Load the images
    image = cv2.imread(image_path)
    background_image = cv2.imread(background_path)
    
    # Convert images to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
    
    # Calculate the absolute difference between the two grayscale images
    difference = cv2.absdiff(gray_background, gray_image)
    cv2.imwrite('difference_image.png', difference)
    
    # Apply a threshold to binarize the difference image
    _, thresh = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)
    cv2.imwrite('thresholded_difference.png', thresh)
    
    # Define the kernel for morphological operations
    kernel = np.ones((15,15), np.uint8)
    
    # Apply errode to make the white regions more compact
    errode_mask = cv2.erode(thresh, kernel, iterations=2)
    
       #kernel = np.ones((15,15), np.uint8)
    # Apply the open operation to remove small artifacts
    kernel = np.ones((25,25), np.uint8)
    opened_mask = cv2.morphologyEx(errode_mask, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('opened_mask.png', opened_mask)
    
    # Apply dilation to make the white regions more compact
    #dilated_mask = cv2.dilate(errode_mask, kernel, iterations=2)
    #cv2.imwrite('dilated_mask.png', dilated_mask)
    
    kernel = np.ones((105,105), np.uint8)
    # Apply the close operation to fill gaps and remove noise
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('closed_mask.png', closed_mask)
    
    #kernel = np.ones((15,15), np.uint8)
    # Apply the open operation to remove small artifacts
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
    cv2.imwrite('opened_mask.png', opened_mask)
    
    # Return the final mask after open and close operations
    return opened_mask



import cv2
import numpy as np

def generate_color_mask(image_path, background_path):
    # Load the images
    image = cv2.imread(image_path)
    background_image = cv2.imread(background_path)
    
    # Convert images to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_background = cv2.cvtColor(background_image, cv2.COLOR_BGR2HSV)
    
    # Calculate the absolute difference between the two HSV images
    difference = cv2.absdiff(hsv_background, hsv_image)
    
    # Separate the Hue, Saturation, and Value channels
    h_diff, s_diff, v_diff = cv2.split(difference)
    
    # Threshold the differences to create masks for each channel
    _, h_mask = cv2.threshold(h_diff, 50, 255, cv2.THRESH_BINARY)
    _, s_mask = cv2.threshold(s_diff, 50, 255, cv2.THRESH_BINARY)
    _, v_mask = cv2.threshold(v_diff, 50, 255, cv2.THRESH_BINARY)
    
    # Combine the masks using logical OR to get the final mask
    combined_mask = cv2.bitwise_or(h_mask, cv2.bitwise_or(s_mask, v_mask))
    
    # Define the kernel for morphological operations
    kernel = np.ones((5,5), np.uint8)
    
    # Apply the close operation to fill gaps and remove noise
    closed_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply the open operation to remove small artifacts
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)
    
    # Mask the original image to highlight only the new colors
    result_image = cv2.bitwise_and(image, image, mask=opened_mask)
    
    # Save the intermediate and final results
    cv2.imwrite('h_diff.png', h_diff)
    cv2.imwrite('s_diff.png', s_diff)
    cv2.imwrite('v_diff.png', v_diff)
    cv2.imwrite('combined_mask.png', combined_mask)
    cv2.imwrite('closed_mask.png', closed_mask)
    cv2.imwrite('opened_mask.png', opened_mask)
    cv2.imwrite('result_image.png', result_image)
    
    # Return the final result image
    return result_image

# Calea către imagini
image_path = f"cadre/{3}.jpg"
background_path = f"cadre/{1}.jpg"

# Generăm masca rafinată
refined_mask = generate_color_mask(image_path, background_path)


# Opțional, afișăm masca
import matplotlib.pyplot as plt

plt.imshow(refined_mask, cmap='gray')
plt.title('Refined Mask')
plt.axis('off')
plt.show()
