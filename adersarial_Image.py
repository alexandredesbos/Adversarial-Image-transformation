import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Load a pre-trained model (Here we use the MobileNetV2 model)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def preprocess_image(image_path):
    """ Load and preprocess the image. """
    # Load the image
    response = requests.get(image_path)
    img = Image.open(BytesIO(response.content))
    img = img.resize((224, 224))
    img = img.convert('RGB')

    # Preprocess for model input
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def create_adversarial_pattern(input_image, input_label):
    """ Create the adversarial pattern using the Fast Gradient Sign Method. """
    input_image = tf.convert_to_tensor(input_image)
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = tf.keras.losses.sparse_categorical_crossentropy(input_label, prediction)

    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

def display_images(image, description):
    """ Display the image with a description. """
    plt.figure()
    plt.imshow(image[0] * 0.5 + 0.5)
    plt.title(description)
    plt.show()

# URL of the image ( Here a pack of mushrooms )
image_path = "https://shop.tonyandmarks.com.au/cdn/shop/products/mushroompack_1200x1200.png?v=1590534548" 

# Preprocess the image
original_image = preprocess_image(image_path)

# Predict the class of the original image
original_pred = model.predict(original_image)
original_class = np.argmax(original_pred[0])

# Create the adversarial image
perturbations = create_adversarial_pattern(original_image, np.array([original_class]))
epsilon = 0.5  # Perturbation amount
adv_image = original_image + epsilon * perturbations

# Display the original and adversarial images
display_images(original_image, 'Original Image')
display_images(adv_image, 'Adversarial Image')

# Predict the class of the adversarial image
adv_pred = model.predict(adv_image)
adv_class = np.argmax(adv_pred[0])

print(f"Original Prediction: {original_class}, Adversarial Prediction: {adv_class}")

