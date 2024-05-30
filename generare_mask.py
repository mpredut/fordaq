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
    result = np.zeros_like(diff, dtype=np.uint8)
    result[blurred_diff > threshold_value] = 255

    return result

def extract_new_object(background_path, foreground_path, output_path, window_size=80, threshold_value=30, binarization_threshold=127):
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
   
    # Apply local thresholding
    mask = apply_local_threshold(diff, window_size, threshold_value)
    
    # Find connected components to create a bounding box mask
    num_labels, labels_im = cv2.connectedComponents(mask)
    if num_labels > 1:
        x_min, y_min, x_max, y_max = float('inf'), float('inf'), float('-inf'), float('-inf')
        for label in range(1, num_labels):
            y, x = np.where(labels_im == label)
            if len(x) > 0 and len(y) > 0:
                x_min = min(x_min, np.min(x))
                y_min = min(y_min, np.min(y))
                x_max = max(x_max, np.max(x))
                y_max = max(y_max, np.max(y))
        
        # Create a new mask with the bounding box
        new_mask = np.zeros_like(mask)
        new_mask[y_min:y_max+1, x_min:x_max+1] = 255
    else:
        new_mask = mask

    # Resize mask to original image size
    mask_resized = cv2.resize(new_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    # Save intermediate mask image
    intermediate_mask_path = f"{output_path}_intermediate_mask.png"
    cv2.imwrite(intermediate_mask_path, mask_resized)
    
    # Apply the bounding box mask to the original foreground image
    masked_image = cv2.bitwise_and(original_foreground, original_foreground, mask=mask_resized)
    
    # Convert masked image to grayscale for contour detection
    masked_gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    
    # Apply fixed threshold to binarize the grayscale image
    _, masked_binary = cv2.threshold(masked_gray, binarization_threshold, 255, cv2.THRESH_BINARY)
    
    # Save the binary image for debugging
    cv2.imwrite("masked_binary.png", masked_binary)
    
    # Find contours in the masked image
    contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Debug: Print the number of contours found
    print(f"Number of contours found: {len(contours)}")
    
    # Create an image to draw all contours
    all_contours_image = np.zeros_like(masked_binary)
    cv2.drawContours(all_contours_image, contours, -1, 255, thickness=1)
    
    # Save the image with all contours
    all_contours_path = f"{output_path}_all_contours.png"
    cv2.imwrite(all_contours_path, all_contours_image)
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Apply convex hull to get a convex shape
    hull = cv2.convexHull(largest_contour)
    
    # Draw the convex hull on a new binary mask
    contour_mask = np.zeros_like(masked_binary)
    cv2.drawContours(contour_mask, [hull], -1, 255, thickness=cv2.FILLED)
    
    # Save the new binary mask with the convex hull
    contour_mask_path = f"{output_path}_contour_mask.png"
    cv2.imwrite(contour_mask_path, contour_mask)
    
    # Apply the new contour mask to the original foreground image
    contour_rgb = cv2.cvtColor(contour_mask, cv2.COLOR_GRAY2RGB)
    extracted_object = cv2.bitwise_and(original_foreground, contour_rgb)
    
    # Save the final result image with the extracted object
    result_with_contour_path = f"{output_path}_extracted_object.png"
    cv2.imwrite(result_with_contour_path, cv2.cvtColor(extracted_object, cv2.COLOR_RGB2BGR))
    
    return result_with_contour_path, intermediate_mask_path, contour_mask_path, all_contours_path

# Calea către imagini
image_path = f"cadre/{10}.jpg"
background_path = f"cadre/{1}.jpg"

for i in range(2, 20):
    image_path = f"cadre/{i}.jpg"
    
    # Generăm masca rafinată
    result_image_path, intermediate_mask_path, contour_mask_path, all_contours_path = extract_new_object(background_path, image_path, f"output_{i}", binarization_threshold=127)
    
    # Display the results
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 5, 1)
    plt.title(f"Foreground Image {i}")
    plt.imshow(load_image(image_path))
    plt.axis('off')
    
    plt.subplot(1, 5, 2)
    plt.title("Intermediate Mask")
    mask_image = load_image(intermediate_mask_path)
    plt.imshow(mask_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 5, 3)
    plt.title("All Contours")
    all_contours_image = cv2.imread(all_contours_path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(all_contours_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 5, 4)
    plt.title("Largest Convex Contour Mask")
    contour_mask_image = load_image(contour_mask_path)
    plt.imshow(contour_mask_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 5, 5)
    plt.title("Extracted New Object")
    result_image = load_image(result_image_path)
    plt.imshow(result_image)
    plt.axis('off')
    
    plt.show()
