import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = '/mnt/data/4.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Watershed Algorithm
def watershed_segmentation(image):
    # Convert image to color for visualization
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Apply thresholding
    ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0 but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 0] = 0

    # Apply the Watershed algorithm
    markers = cv2.watershed(image_color, markers)
    image_color[markers == -1] = [255, 0, 0]

    # Create a mask where markers are >1 (foreground regions)
    mask = np.zeros_like(image)
    mask[markers > 1] = 255

    return mask

# Region Growing
def region_growing(image):
    # Seed point
    seed = (image.shape[0]//2, image.shape[1]//2)  # Center of the image

    # Create a mask initialized with zeros
    mask = np.zeros_like(image)

    # Parameters for region growing
    threshold = 10  # Intensity difference threshold
    queue = [seed]
    while queue:
        x, y = queue.pop(0)
        if mask[x, y] == 0 and abs(int(image[x, y]) - int(image[seed])) < threshold:
            mask[x, y] = 255
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                    queue.append((nx, ny))

    return mask

# Apply watershed segmentation
watershed_result = watershed_segmentation(image)

# Apply region growing
region_growing_result = region_growing(image)

# Display the results
plt.figure(figsize=(15, 10))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Watershed Segmentation")
plt.imshow(watershed_result, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Region Growing Segmentation")
plt.imshow(region_growing_result, cmap='gray')
plt.axis('off')

plt.tight_layout()
comparison_image_path = '/mnt/data/comparison_watershed_region_growing_corrected.png'
plt.savefig(comparison_image_path)
plt.show()

comparison_image_path
