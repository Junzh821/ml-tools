import numpy as np
import os
import shutil
import time

from subprocess import call
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.preprocessing import image
from tensorflow_serving_python.client import TFClient

from keras_tools.keras_to_tensorflow import KerasToTensorflow


def load_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size[0:2])
    x = image.img_to_array(img)
    # TODO: x = preprocessing(x)
    return np.expand_dims(x, axis=0)


def test_convert_imagenet_inception_v3():
    model_path = '.cache/models/inception_v3.h5'
    tf_model_dir = '.cache/models/tf/inception_v3'

    target_size = (299, 299, 3)
    if not os.path.exists(model_path):
        weights_path = '.cache/weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
        model = InceptionV3(weights='imagenet', include_top=False, input_shape=target_size)
        model.load_weights(weights_path)
        model.save(model_path)

    if os.path.exists(tf_model_dir):
        shutil.rmtree(tf_model_dir)

    KerasToTensorflow.convert(model_path, tf_model_dir)

    assert os.path.exists(tf_model_dir)
    assert os.path.exists(tf_model_dir + '/variables')
    assert os.path.exists(tf_model_dir + '/variables/variables.data-00000-of-00001')
    assert os.path.exists(tf_model_dir + '/variables/variables.index')
    assert os.path.exists(tf_model_dir + '/saved_model.pb')

    call(['docker-compose', 'restart', 'inception_serving'])
    time.sleep(3)

    client = TFClient('localhost', '9001')
    img = load_image('tests/fixtures/files/cat.jpg', target_size)
    assert client.make_prediction(img, 'image', timeout=10, name='inception')


def test_convert_imagenet_mobilenet():
    model_path = '.cache/models/mobilenet.h5'
    tf_model_dir = '.cache/models/tf/mobilenet'

    target_size = (224, 224, 3)
    if not os.path.exists(model_path):
        weights_path = '.cache/weights/mobilenet_1_0_224_tf_no_top.h5'
        model = MobileNet(weights='imagenet', include_top=False, input_shape=target_size)
        model.load_weights(weights_path)
        model.save(model_path)

    if os.path.exists(tf_model_dir):
        shutil.rmtree(tf_model_dir)

    KerasToTensorflow.convert(model_path, tf_model_dir)

    assert os.path.exists(tf_model_dir)
    assert os.path.exists(tf_model_dir + '/variables')
    assert os.path.exists(tf_model_dir + '/variables/variables.data-00000-of-00001')
    assert os.path.exists(tf_model_dir + '/variables/variables.index')
    assert os.path.exists(tf_model_dir + '/saved_model.pb')

    call(['docker-compose', 'restart', 'mobilenet_serving'])
    time.sleep(3)

    client = TFClient('localhost', '9002')
    img = load_image('tests/fixtures/files/cat.jpg', target_size)
    assert client.make_prediction(img, 'image', timeout=10, name='mobilenet')


def test_convert_imagenet_resnet50():
    model_path = '.cache/models/resnet50.h5'
    tf_model_dir = '.cache/models/tf/resnet50'

    target_size = (224, 224, 3)
    if not os.path.exists(model_path):
        weights_path = '.cache/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        model = ResNet50(weights='imagenet', include_top=False, input_shape=target_size)
        model.load_weights(weights_path)
        model.save(model_path)

    if os.path.exists(tf_model_dir):
        shutil.rmtree(tf_model_dir)

    KerasToTensorflow.convert(model_path, tf_model_dir)

    assert os.path.exists(tf_model_dir)
    assert os.path.exists(tf_model_dir + '/variables')
    assert os.path.exists(tf_model_dir + '/variables/variables.data-00000-of-00001')
    assert os.path.exists(tf_model_dir + '/variables/variables.index')
    assert os.path.exists(tf_model_dir + '/saved_model.pb')

    call(['docker-compose', 'restart', 'resnet50_serving'])
    time.sleep(3)

    client = TFClient('localhost', '9003')
    img = load_image('tests/fixtures/files/cat.jpg', target_size)
    assert client.make_prediction(img, 'image', timeout=10, name='resnet50')


def test_convert_imagenet_xception():
    model_path = '.cache/models/xception.h5'
    tf_model_dir = '.cache/models/tf/xception'

    target_size = (299, 299, 3)
    if not os.path.exists(model_path):
        weights_path = '.cache/weights/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'
        model = Xception(weights='imagenet', include_top=False, input_shape=target_size)
        model.load_weights(weights_path)
        model.save(model_path)

    if os.path.exists(tf_model_dir):
        shutil.rmtree(tf_model_dir)

    KerasToTensorflow.convert(model_path, tf_model_dir)

    assert os.path.exists(tf_model_dir)
    assert os.path.exists(tf_model_dir + '/variables')
    assert os.path.exists(tf_model_dir + '/variables/variables.data-00000-of-00001')
    assert os.path.exists(tf_model_dir + '/variables/variables.index')
    assert os.path.exists(tf_model_dir + '/saved_model.pb')

    call(['docker-compose', 'restart', 'xception_serving'])
    time.sleep(3)

    client = TFClient('localhost', '9004')
    img = load_image('tests/fixtures/files/cat.jpg', target_size)
    assert client.make_prediction(img, 'image', timeout=10, name='xception')
