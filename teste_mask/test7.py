import cv2
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer

def load_image(file_path):
    """Load an image from file path."""
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def generate_mask(image_path, output_path):
    """Generate object mask using Mask R-CNN."""
    # Load image
    image = load_image(image_path)

    # Configure detectron2
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)

    # Perform inference on the image
    outputs = predictor(image)

    # Get the mask for the detected objects
    masks = outputs["instances"].pred_masks.cpu().numpy()

    # Create a combined mask
    combined_mask = np.zeros_like(image[:, :, 0], dtype=np.uint8)
    for mask in masks:
        combined_mask[mask] = 255

    # Save the mask
    mask_path = f"{output_path}_mask.png"
    cv2.imwrite(mask_path, combined_mask)

    return mask_path, combined_mask

# Calea către imagine
image_path = "path_to_image.jpg"

# Generăm și salvăm masca obiectului
mask_image_path, mask_image = generate_mask(image_path, "output")

# Display the results
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(load_image(image_path))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Generated Mask")
plt.imshow(mask_image, cmap='gray')
plt.axis('off')

plt.show()
