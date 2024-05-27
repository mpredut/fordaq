import cv2
import numpy as np


# """Expand white regions of the mask horizontally by 'offset' 
# pixels on each side without exceeding the image dimensions."""
def safely_expand_mask(image, mask, offset):
    img_height, img_width = image.shape[:2]  # Dimensions of the associated image
    mask_height, mask_width = mask.shape
    
    # Create a new mask of the same height and width as the image, filled with zeros
    new_mask = np.zeros((img_height, img_width), dtype=mask.dtype)
    
    # Iterate over each row in the mask
    for row in range(mask_height):
        # Find indices of all white pixels
        white_indices = np.where(mask[row] == 255)[0]
        if white_indices.size > 0:
            # Calculate the start and end points for the white region expansion
            start_index = max(white_indices[0] - offset, 0)
            end_index = min(white_indices[-1] + offset + 1, img_width)
            # Set the expanded region to white
            new_mask[row, start_index:end_index] = 255
    
    return new_mask


# """Shift the image horizontally by num_cols.
# Positive num_cols shifts to the right, negative num_col shifts to the left."""
def shift_image(image, num_cols):
    if num_cols > 0:
        # Shift to the right
        shifted_image = np.roll(image, num_cols, axis=1)
        shifted_image[:, :num_cols] = 0
    else:
        # Shift to the left
        shifted_image = np.roll(image, num_cols, axis=1)
        shifted_image[:, num_cols:] = 0
    return shifted_image

   
    
    
# Function to calculate the vertical shift based on the end of the mask
def calculate_vertical_shift(mask1, mask2):
    # Sum across rows to identify rows that are fully black (sum = 0) or have white (sum > 0)
    row_sums1 = np.sum(mask1 > 0, axis=1)
    row_sums2 = np.sum(mask2 > 0, axis=1)

    # Identify the last row that contains white pixels (non-zero sum)
    y_end_1 = np.where(row_sums1 > 0)[0].max() if np.any(row_sums1 > 0) else None
    y_end_2 = np.where(row_sums2 > 0)[0].max() if np.any(row_sums2 > 0) else None

    # Calculate the shift only if both masks have detectable ends
    if y_end_1 is not None and y_end_2 is not None:
        shift = y_end_2 - y_end_1
        return shift
    else:
        return None  # This will handle cases where masks do not have a clear end

 
"""Load mask and image files from specified paths and expand masks safely.""" 
def load_images_and_masks(mask_paths, image_paths, offset=200):

    images = [cv2.imread(image_path) for image_path in image_paths]
    masks = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in mask_paths]
    
    # Check if image and mask lists are of the same length
    if len(images) != len(masks):
        raise ValueError("The number of images and masks must be the same.")
    
    # Apply mask expansion to each mask according to its corresponding image
    expanded_masks = []
    for image, mask in zip(images, masks):
        if image is not None and mask is not None:
            expanded_mask = safely_expand_mask(image, mask, offset)
            expanded_masks.append(expanded_mask)
        else:
            # Handle cases where image or mask could not be loaded
            expanded_masks.append(None)
    
    return images, expanded_masks
    
"""Extract a specified region from an image using a mask, or return None if mask is empty."""
def extract_region(image, mask):
    if mask is None or np.sum(mask) <= 100:  # Check if mask is empty
        return None
    return cv2.bitwise_and(image, image, mask=mask)

"""Combine two image regions into one with a blurred transition."""
def combine_regions(upper_region, lower_region, transition_point, blur_width=20):
    combined_image = np.vstack((upper_region, lower_region))
    
    # Smooth the transition zone
    transition_zone_start = transition_point - blur_width // 2
    transition_zone_end = transition_point + blur_width // 2
    transition_zone = combined_image[transition_zone_start:transition_zone_end]
    blurred_transition = cv2.GaussianBlur(transition_zone, (0, 0), sigmaX=15, sigmaY=15)
    combined_image[transition_zone_start:transition_zone_end] = blurred_transition
    
    return combined_image

def process_images(mask_paths, image_paths, shift_value):
    images, masks = load_images_and_masks(mask_paths, image_paths)
    image_scandura = []

    for image, mask in zip(images, masks):
        extracted_image = extract_region(image, mask)
        if extracted_image is not None:
            image_scandura.append(extracted_image)
    
    if not image_scandura:
        return None  # Return None if no images were extracted

    # Initialize final_image with the first valid extracted image
    final_image = image_scandura[0]

    i = 0
    for extracted_image in image_scandura[1:]:
        new_top_region = extracted_image[:shift_value]  # Extract the top part of the current region
        #new_top_region = shift_image(new_top_region, 0)
        final_image = combine_regions(new_top_region, final_image, shift_value)
        #cv2.imwrite(f"combined_image{i}.jpg", final_image)
        i = i + 1

    return final_image

# Main code:
mask_paths = [f"mask/mask_{i}.png" for i in range(1, 19)]
image_paths = [f"cadre/{i}.jpg" for i in range(1, 19)]
shift_value = 0


masks = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in mask_paths]
# Calculate vertical shifts using the adjusted function
vertical_shifts = []
for i in range(1, 4):
    shift = calculate_vertical_shift(masks[i-1], masks[i])
    if shift is not None:
        vertical_shifts.append(shift)

# Calculate the average shift, excluding None values
if vertical_shifts:
    average_vertical_shift = np.mean(vertical_shifts)
else:
    average_vertical_shift = None

print("Vertical shifts:", vertical_shifts)
print("Average vertical shift:", average_vertical_shift)

shift_value = int(average_vertical_shift)  

final_result = process_images(mask_paths, image_paths, shift_value)

rotated_image = cv2.rotate(final_result, cv2.ROTATE_90_CLOCKWISE)

cv2.imwrite('imagine_rotita.jpg', rotated_image)
    
    


