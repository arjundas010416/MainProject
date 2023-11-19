import cv2
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

model = MobileNetV2(weights='imagenet')
def detect_pest(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions)

    # Check if a pest is detected based on the top prediction
    top_prediction = decoded_predictions[0][0]
    if "pest" in top_prediction[1].lower():
        return True
    else:
        return False

if __name__ == "__main__":
    image_path = "pest1.jpeg"
    is_pest = detect_pest(image_path)

    if is_pest:
        print("Pest detected in the image.")
    else:
        print("No pest detected in the image.")
