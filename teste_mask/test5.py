import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import percentile_filter







def load_image(file_path):
    """Load an image from file path."""
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def resize_image(image, size=(256, 256)):
    """Resize image to the given size."""
    return cv2.resize(image, size)

def apply_local_threshold(diff, window_size, threshold_value):
    if window_size % 2 == 0:
        window_size += 1
    """Apply local thresholding to classify pixels into background, edge, and object."""
    blurred_diff = cv2.GaussianBlur(diff, (window_size, window_size), 0)
    percentile = 70
    #filtered_diff = percentile_filter(diff, percentile, size=window_size)
    # Classify pixels based on the blurred image
    result = np.zeros_like(diff, dtype=np.uint8)
    result[blurred_diff > threshold_value] = 255

    return result

def extract_new_object(background_path, foreground_path, output_path, window_size=80, threshold_value=30):
    """Extract the new object from the foreground image."""
    # Load and resize images
    original_foreground = load_image(foreground_path)
    original_size = (original_foreground.shape[1], original_foreground.shape[0])
    background = resize_image(load_image(background_path))
    foreground = resize_image(original_foreground)
    
    # Convert images to grayscale
    background_gray = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)
    foreground_gray = cv2.cvtColor(foreground, cv2.COLOR_RGB2GRAY)
    
    # Subtract background from foreground
    diff = cv2.absdiff(foreground_gray, background_gray)
   
    # Apply morphological operations to clean up the mask
    #kernel = np.ones((5,5), np.uint8)
    #diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
    #diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
    
    # Apply local thresholding
    mask = apply_local_threshold(diff, window_size, threshold_value)
    
    
    # Resize mask to original image size
    mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    # Save intermediate mask image
    intermediate_mask_path = f"{output_path}_mask.png"
    cv2.imwrite(intermediate_mask_path, mask_resized)
    
    # Convert the mask to 3 channels to apply it to the RGB image
    mask_rgb = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2RGB)
    
    # Create an image with the new object (foreground) using the mask
    new_object = cv2.bitwise_and(original_foreground, mask_rgb)
    
    # Save the final result image
    result_path = f"{output_path}_new_object.png"
    cv2.imwrite(result_path, cv2.cvtColor(new_object, cv2.COLOR_RGB2BGR))
    
    return result_path, mask_resized

# Calea către imagini
image_path = f"cadre/{10}.jpg"
background_path = f"cadre/{1}.jpg"





for i in range(2, 20):
    image_path = f"cadre/{i}.jpg"
    
    # Generăm masca rafinată
    result_image_path, mask_image = extract_new_object(background_path, image_path, f"output_{i}")
    
    # Display the results
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 3, 1)
    plt.title(f"Foreground Image {i} with New Object")
    plt.imshow(load_image(image_path))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Intermediate Mask")
    plt.imshow(mask_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Extracted New Object")
    result_image = load_image(result_image_path)
    plt.imshow(result_image)
    plt.axis('off')
    
    plt.show()
    