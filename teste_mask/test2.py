import cv2
import numpy as np
import os
import cv2
import numpy as np
import os
import cv2
import numpy as np
import os


def load_images_and_masks(image_paths, mask_paths):
    images = []
    masks = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        images.append(image)
        masks.append(mask)
    return images, masks

def extract_masked_region(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)


def load_masks(mask_paths):
    masks = []
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        masks.append(mask)
    return masks

def calculate_vertical_shifts(masks):
    shifts = []
    for i in range(len(masks) - 1):
        mask1 = masks[i]
        mask2 = masks[i + 1]

        result = cv2.matchTemplate(mask2, mask1, cv2.TM_CCORR_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        # max_loc gives the top-left corner of the best match in mask2
        vertical_shift = max_loc[1]
        shifts.append(vertical_shift)

    return shifts

def average_vertical_shift(shifts):
    return int(np.mean(shifts))

def find_best_overlap_shift(region1_np, region2_np, estimated_shift):
    region1_gray = cv2.cvtColor(region1_np, cv2.COLOR_BGR2GRAY)
    region2_gray = cv2.cvtColor(region2_np, cv2.COLOR_BGR2GRAY)

    min_height = min(region1_gray.shape[0], region2_gray.shape[0])
    max_correlation = -np.inf
    best_shift = 0

    for shift in range(max(0, estimated_shift - 10), min(min_height, estimated_shift + 10)):
        overlap_region1 = region1_gray[-(shift + 1):]
        overlap_region2 = region2_gray[:shift + 1]

        correlation = np.sum(overlap_region1 * overlap_region2)
        
        if correlation > max_correlation:
            max_correlation = correlation
            best_shift = shift + 1

    return best_shift

def mark_overlap_region(region1, region2, overlap_height):
    marked_region1 = region1.copy()
    marked_region2 = region2.copy()

    # Mark the overlap region on region1
    cv2.rectangle(marked_region1, (0, marked_region1.shape[0] - overlap_height), 
                  (marked_region1.shape[1], marked_region1.shape[0]), (0, 255, 0), 3)

    # Mark the overlap region on region2
    cv2.rectangle(marked_region2, (0, 0), 
                  (marked_region2.shape[1], overlap_height), (0, 255, 0), 3)
    
    return marked_region1, marked_region2

def combine_regions(region1_np, region2_np, overlap_height):
    new_region = region2_np[:-overlap_height]
    combined_height = new_region.shape[0] + region1_np.shape[0]
    combined_width = max(new_region.shape[1], region1_np.shape[1])
    
    combined_image = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    combined_image[:new_region.shape[0], :new_region.shape[1]] = new_region
    combined_image[new_region.shape[0]:, :region1_np.shape[1]] = region1_np
    
    return combined_image

def lipire_cadre(image_folder, mask_folder, image_count, output_filename):
    image_paths = [os.path.join(image_folder, f'{i}.jpg') for i in range(4, 4 + image_count)]
    mask_paths = [os.path.join(mask_folder, f'mask_{i}.png') for i in range(1, 1 + image_count)]
    
    images = [cv2.imread(image_path) for image_path in image_paths]
    masks = load_masks(mask_paths)
    
    # Estimate vertical shifts
    shifts = calculate_vertical_shifts(masks)
    estimated_shift = average_vertical_shift(shifts)
    
    image1 = images[0]
    mask1 = masks[0]
    image2 = images[1]
    mask2 = masks[1]
    
    region1 = extract_masked_region(image1, mask1)
    region2 = extract_masked_region(image2, mask2)
    
    # Save the regions as images
    region1_path = os.path.join(image_folder, 'region1.jpg')
    region2_path = os.path.join(image_folder, 'region2.jpg')
    cv2.imwrite(region1_path, region1)
    cv2.imwrite(region2_path, region2)
    
    print("Regions saved")
    
    region1_np = np.array(region1)
    region2_np = np.array(region2)
    
    overlap_height = find_best_overlap_shift(region1_np, region2_np, estimated_shift)
    
    if overlap_height > 0:
        marked_region1, marked_region2 = mark_overlap_region(region1_np, region2_np, overlap_height)
        
        # Save the marked regions
        marked_region1_path = os.path.join(image_folder, 'marked_region1.jpg')
        marked_region2_path = os.path.join(image_folder, 'marked_region2.jpg')
        cv2.imwrite(marked_region1_path, marked_region1)
        cv2.imwrite(marked_region2_path, marked_region2)
        
        combined_image_np = combine_regions(region1_np, region2_np, overlap_height)
        cv2.imwrite(output_filename, combined_image_np)
        print(f"Combined image saved as {output_filename}")
        print(f"Marked region1 saved as {marked_region1_path}")
        print(f"Marked region2 saved as {marked_region2_path}")
    else:
        print("No overlap found between the regions.")

# Example usage
lipire_cadre("cadre", "mask", 2, "scandura_completa_compact.jpg")



# Display the combined image
#combined_image_pil = Image.open('scandura_completa_compact.jpg')
#plt.imshow(combined_image_pil)
#plt.axis('off')
#plt.show()
