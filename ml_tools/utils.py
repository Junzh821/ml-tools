import numpy as np

from keras.preprocessing.image import load_img


def load_image(image_path, target_size=None, preprocess_input=None):
    img = load_img(image_path, target_size=target_size[:2])
    image_data = np.asarray(img, dtype=np.float32)
    image_data = np.expand_dims(image_data, axis=0)
    if preprocess_input:
        image_data = preprocess_input(image_data)
    return image_data
