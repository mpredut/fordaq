import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import tflite_runtime.interpreter as tflite
import kagglehub


def load_and_prepare_image(file_path, target_size):
    """Load and prepare the image for segmentation."""
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0  # Normalize the image to [0, 1]
    img = img.astype(np.float32)
    return img

def segment_image(image_path):
    """Segment the image using a TensorFlow Lite model."""
    model_path = "/home/marius/.cache/kagglehub/models/tensorflow/deeplabv3/tfLite/default/1/1.tflite"

    # Load the TensorFlow Lite model
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Inspect the model's input shape
    input_shape = input_details[0]['shape']
    print(f"Model's expected input shape: {input_shape}")

    # Prepare the image
    image_size = (input_shape[1], input_shape[2])  # Resize the image to match the model's expected input size
    img = load_and_prepare_image(image_path, image_size)
    img_tensor = np.expand_dims(img, axis=0)  # Extend dimensions to include batch size

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], img_tensor)

    # Run inference
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data
    # Use `tensor()` in order to get a pointer to the tensor
    mask = interpreter.get_tensor(output_details[0]['index'])
    mask = np.argmax(mask, axis=-1)
    mask = mask[0]  # Remove batch size

    return img, mask


# Download latest version
path = kagglehub.model_download("tensorflow/deeplabv3/tfLite/default")

print("Path to model files:", path)

# Utilizarea funcției
image_path =  f"cadre/{3}.jpg"
original_image, predicted_mask = segment_image(image_path)

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(predicted_mask, cmap='gray')
plt.title('Segmented Mask')
plt.axis('off')

plt.show()



   
# Calea către imagini
image_path = f"cadre/{3}.jpg"
background_path = f"cadre/{1}.jpg"

# Generăm masca rafinată
refined_mask = generate_mask_with_better_edges(image_path, background_path)

# Opțional, afișăm masca
import matplotlib.pyplot as plt

plt.imshow(refined_mask, cmap='gray')
plt.title('Refined Mask')
plt.axis('off')
plt.show()
