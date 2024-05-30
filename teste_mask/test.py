import cv2
import numpy as np

def lipire_cadre(cadre_path, masti_path, numar):
    cadre = [cv2.imread(f"{cadre_path}/{i}.jpg") for i in range(1, numar + 1)]
    masti = [cv2.imread(f"{masti_path}/mask_{i}.png") for i in range(1, numar + 1)]
    
    # Check if images are loaded correctly
    for idx, img in enumerate(cadre, start=1):
        if img is None:
            print(f"Error: Image at {cadre_path}/{idx}.jpg could not be loaded.")
    for idx, img in enumerate(masti, start=1):
        if img is None:
            print(f"Error: Image at {masti_path}/mask_{idx}.png could not be loaded.")
    
    # If any image is not loaded, exit the function
    if any(img is None for img in cadre + masti):
        return
    
    # Get the dimensions of the cadre images
    inaltime, latime, _ = cadre[0].shape
    
    # Initialize a blank scandura image
    scandura = np.zeros((inaltime, latime, 3), dtype=np.uint8)
    print(f"Initialized blank scandura with shape: {scandura.shape}")

    for i in range(numar):
        cadre[i] = cv2.resize(cadre[i], (latime, inaltime))
        masti[i] = cv2.resize(masti[i], (latime, inaltime))
        print(f"Cadre and mask {i+1} resized to: {cadre[i].shape}")

        masca = masti[i]
        if masca is None:
            print(f"Error: Mask at index {i+1} is None.")
            continue
        
        # Convert mask to grayscale if it has multiple channels
        if len(masca.shape) == 3 and masca.shape[2] == 3:
            masca_gray = cv2.cvtColor(masca, cv2.COLOR_BGR2GRAY)
        else:
            masca_gray = masca
        
        print(f"Mask {i+1} shape: {masca_gray.shape}")

        # Find indices where mask is greater than 0
        y_indices, x_indices = np.where(masca_gray > 0)
        if y_indices.size == 0 or x_indices.size == 0:
            print(f"Warning: No mask found in image at index {i+1}.")
            continue

        # Overlay the pieces from the cadre image onto the scandura image
        for y, x in zip(y_indices, x_indices):
            if masca_gray[y, x] > 0:
                scandura[y, x] = cadre[i][y, x]

        print(f"Processed mask and cadre at index {i+1}")

    # Save the final image
    output_path = "output.jpg"
    success = cv2.imwrite(output_path, scandura)
    if success:
        print(f"Output image saved successfully at {output_path}")
    else:
        print(f"Failed to save the output image at {output_path}")

# Example usage
lipire_cadre("cadre", "mask", 10)
