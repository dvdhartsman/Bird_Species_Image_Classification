
# Beginning of the python script for streamlit
import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model

model = load_model(os.path.join(path, "models", "bird_classifier_1.h5"))

def format_image(img_path):
    img = mpimg.imread(img_path)
    resized_image = tf.image.resize(img, (224,224))
    return resized_image


def get_prediction(resized_image):
    y_pred = model.predict(np.expand_dims(resized_image/255, 0))
    pred_class = np.argmax(y_pred[0])
    class_name = reverse_dict[pred_class]
    if y_pred[0][pred_class] >= .5:
        confidence_level = "High Confidence"
    elif .35< y_pred[0][pred_class] < .5:
        confidence_level = "Moderate Confidence"
    else:
        confidence_level = "Low Confidence"
    print(f"Predicted Bird is: {class_name} with {confidence_level}")
