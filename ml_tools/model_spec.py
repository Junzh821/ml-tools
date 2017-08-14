from keras.applications.inception_v3 import InceptionV3, preprocess_input as inception_v3_preprocess_input
from keras.applications.xception import Xception, preprocess_input as xception_preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input as mobilenet_v1_preprocess_input
from keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input as resnet50_preprocess_input

from .utils import load_image


MODEL_SPECS = {
    'inception_v3': {
        'class': InceptionV3,
        'target_size': (299, 299, 3),
        'preprocess_input': inception_v3_preprocess_input,  # imagenet-alt
        # https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py#L390-L394
    },
    'mobilenet_v1': {
        'class': MobileNet,
        'target_size': (224, 224, 3),
        'preprocess_input': mobilenet_v1_preprocess_input,  # imagenet-alt
        # https://github.com/fchollet/keras/blob/master/keras/applications/mobilenet.py#L86-L90
    },
    'xception': {
        'class': Xception,
        'target_size': (299, 299, 3),
        'preprocess_input': xception_preprocess_input,  # imagenet-alt
        # https://github.com/fchollet/keras/blob/master/keras/applications/xception.py#L263-L267
    },
    'resnet50': {
        'class': ResNet50,
        'target_size': (224, 224, 3),
        'preprocess_input': resnet50_preprocess_input,  # imagenet
        # https://github.com/fchollet/keras/blob/master/keras/applications/resnet50.py#L32
    },
    'vgg16': {
        'class': VGG16,
        'target_size': (224, 224, 3),
        'preprocess_input': vgg16_preprocess_input,  # imagenet
        # https://github.com/fchollet/keras/blob/master/keras/applications/vgg16.py#L27
    },
    'vgg19': {
        'class': VGG19,
        'target_size': (224, 224, 3),
        'preprocess_input': vgg19_preprocess_input,  # imagenet
        # https://github.com/fchollet/keras/blob/master/keras/applications/vgg19.py#L27
    },
}


# preprocessing functions
#
# "imagenet"
# https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py#L11-L39
# def preprocessing(x):
#   x -= [<dataset per-channel means>]
#   x /= 255.
#   x *= 2.
#   return x
#
# "imagenet-alt"
# https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py#L390-L394
# def preprocess_input(x):
#   x /= 255.
#   x -= 0.5
#   x *= 2.
#   return x


class ModelSpec(object):

    def __init__(self, name, klass, target_size, preprocess_input):
        self.name = name
        self.klass = klass
        self.target_size = target_size
        self.preprocess_input = preprocess_input

    def load_image(self, path):
        return load_image(path, self.target_size, self.preprocess_input)


def get_model_spec(name):
    config = MODEL_SPECS.get(name)
    return ModelSpec(name, config['class'], config['target_size'], config['preprocess_input'])
