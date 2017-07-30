import pytest
import os
import shutil
import time

from subprocess import call
from urllib.request import urlopen
from urllib.error import URLError
from http.client import RemoteDisconnected
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from tensorflow_serving_python.client import TFClient

from keras_tools.keras_to_tensorflow import KerasToTensorflow


def test_convert_imagenet_inception_v3():
    model_path = '.cache/models/inception_v3.h5'
    tf_model_dir = '.cache/models/tf/inception_v3'

    if not os.path.exists(model_path):
        target_size = (299,299, 3)
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

    time.sleep(1)
    call(['docker-compose', 'restart', 'inception_serving'])
    time.sleep(2)

    client = TFClient('localhost', '9001')
    data = open('tests/fixtures/files/cat.jpg', 'rb').read()
    assert client.make_prediction(data, timeout=10, name='inception')


def test_convert_imagenet_mobilenet():
    model_path = '.cache/models/mobilenet.h5'
    tf_model_dir = '.cache/models/tf/mobilenet'

    if not os.path.exists(model_path):
        target_size = (224,224,3)
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

    time.sleep(1)
    call(['docker-compose', 'restart', 'mobilenet_serving'])
    time.sleep(1)

    client = TFClient('localhost', '9002')
    data = open('tests/fixtures/files/cat.jpg', 'rb').read()
    assert client.make_prediction(data, timeout=10, name='mobilenet')


def test_convert_imagenet_resnet50():
    model_path = '.cache/models/resnet50.h5'
    tf_model_dir = '.cache/models/tf/resnet50'

    if not os.path.exists(model_path):
        target_size = (224,224,3)
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

    time.sleep(1)
    call(['docker-compose', 'restart', 'resnet50_serving'])
    time.sleep(2)

    client = TFClient('localhost', '9003')
    data = open('tests/fixtures/files/cat.jpg', 'rb').read()
    assert client.make_prediction(data, timeout=10, name='resnet50')
