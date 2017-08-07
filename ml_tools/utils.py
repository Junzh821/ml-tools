import numpy as np

from PIL import Image

from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_v3_preprocess_input
from keras.applications.xception import Xception, preprocess_input as xception_preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input as mobilenet_v1_preprocess_input
from keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess_input


MODEL_SPECS = {
    'inception_v3': {
        'class': InceptionV3,
        'target_size': (299, 299, 3),
        'preprocess_input': inception_v3_preprocess_input,
    },
    'mobilenet_v1': {
        'class': MobileNet,
        'target_size': (224, 224, 3),
        'preprocess_input': mobilenet_v1_preprocess_input,
    },
    'resnet50': {
        'class': ResNet50,
        'target_size': (224, 224, 3),
        'preprocess_input': resnet50_preprocess_input,
    },
    'xception': {
        'class': Xception,
        'target_size': (299, 299, 3),
        'preprocess_input': xception_preprocess_input,
    },
    'vgg16': {
        'class': VGG16,
        'target_size': (224, 224, 3),
        'preprocess_input': vgg16_preprocess_input,
    },
    'vgg19': {
        'class': VGG19,
        'target_size': (224, 224, 3),
        'preprocess_input': vgg19_preprocess_input,
    },
}


def load_image(image_path, target_size=None, preprocess_input=None):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    if target_size:
        width_height = (target_size[0], target_size[1])
        if img.size != width_height:
            img = img.resize(width_height)
    image_data = np.asarray(img, dtype=np.float32)
    if preprocess_input:
        image_data = preprocess_input(image_data)
    image_data = np.expand_dims(image_data, axis=0)
    return image_data
