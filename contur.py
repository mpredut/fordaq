# Function to detect contours using Prewitt edge detection followed by findContours
def detect_contours_prewitt(image):
    """Detect contours using Prewitt edge detection followed by findContours."""
    # Apply Prewitt edge detection
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    prewittx = cv2.filter2D(image, cv2.CV_32F, kernelx)
    prewitty = cv2.filter2D(image, cv2.CV_32F, kernely)
    prewitt_edges = cv2.magnitude(prewittx, prewitty)
    prewitt_edges = np.uint8(prewitt_edges)
    
    # Apply thresholding to get a binary image from Prewitt edges
    _, binary_prewitt = cv2.threshold(prewitt_edges, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours on the Prewitt edges
    contours, _ = cv2.findContours(binary_prewitt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, binary_prewitt

# Re-run the Sobel, Laplacian, Prewitt, and Scharr edge detection with contour drawing

# Detect contours using Laplacian edge detection
contours_laplacian, laplacian_edges = detect_contours_laplacian(image)
contour_image_laplacian = np.zeros_like(image)
cv2.drawContours(contour_image_laplacian, contours_laplacian, -1, (255), 2)

# Detect contours using Prewitt edge detection
contours_prewitt, prewitt_edges = detect_contours_prewitt(image)
contour_image_prewitt = np.zeros_like(image)
cv2.drawContours(contour_image_prewitt, contours_prewitt, -1, (255), 2)

# Detect contours using Scharr edge detection
contours_scharr, scharr_edges = detect_contours_scharr(image)
contour_image_scharr = np.zeros_like(image)
cv2.drawContours(contour_image_scharr, contours_scharr, -1, (255), 2)

# Plotting the results
plt.figure(figsize=(15, 20))

plt.subplot(4, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(4, 2, 2)
plt.title("Contours Direct")
plt.imshow(contour_image_direct, cmap='gray')
plt.axis('off')

plt.subplot(4, 2, 3)
plt.title("Canny Edges")
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(4, 2, 4)
plt.title("Contours after Canny")
plt.imshow(contour_image_canny, cmap='gray')
plt.axis('off')

plt.subplot(4, 2, 5)
plt.title("Laplacian Edges")
plt.imshow(laplacian_edges, cmap='gray')
plt.axis('off')

plt.subplot(4, 2, 6)
plt.title("Contours after Laplacian")
plt.imshow(contour_image_laplacian, cmap='gray')
plt.axis('off')

plt.subplot(4, 2, 7)
plt.title("Prewitt Edges")
plt.imshow(prewitt_edges, cmap='gray')
plt.axis('off')

plt.subplot(4, 2, 8)
plt.title("Contours after Prewitt")
plt.imshow(contour_image_prewitt, cmap='gray')
plt.axis('off')

plt.subplot(4, 2, 9)
plt.title("Scharr Edges")
plt.imshow(scharr_edges, cmap='gray')
plt.axis('off')

plt.subplot(4, 2, 10)
plt.title("Contours after Scharr")
plt.imshow(contour_image_scharr, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Save the images to files
output_paths.update({
    "Laplacian_Edges": "/mnt/data/laplacian_edges.png",
    "Contours_After_Laplacian": "/mnt/data/contours_after_laplacian.png",
    "Prewitt_Edges": "/mnt/data/prewitt_edges.png",
    "Contours_After_Prewitt": "/mnt/data/contours_after_prewitt.png",
    "Scharr_Edges": "/mnt/data/scharr_edges.png",
    "Contours_After_Scharr": "/mnt/data/contours_after_scharr.png"
})

cv2.imwrite(output_paths["Laplacian_Edges"], laplacian_edges)
cv2.imwrite(output_paths["Contours_After_Laplacian"], contour_image_laplacian)
cv2.imwrite(output_paths["Prewitt_Edges"], prewitt_edges)
cv2.imwrite(output_paths["Contours_After_Prewitt"], contour_image_prewitt)
cv2.imwrite(output_paths["Scharr_Edges"], scharr_edges)
cv2.imwrite(output_paths["Contours_After_Scharr"], contour_image_scharr)

output_paths &#8203;:citation[【oaicite:0】]&#8203;
