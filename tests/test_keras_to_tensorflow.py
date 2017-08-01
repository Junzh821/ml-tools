import os
import time
import shutil
import numpy as np

from subprocess import call
from keras.applications.inception_v3 import InceptionV3
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.preprocessing import image
from tensorflow_serving_python.client import TFClient
from grpc.framework.interfaces.face.face import AbortionError

from keras_tools.keras_to_tensorflow import KerasToTensorflow


MODELS = {
    'inception': {
        'class': InceptionV3,
        'weights': 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'size': (299, 299, 3),
        'port': 9001,
    },
    'mobilenet': {
        'class': MobileNet,
        'weights': 'mobilenet_1_0_224_tf_no_top.h5',
        'size': (224, 224, 3),
        'port': 9002,
    },
    'resnet50': {
        'class': ResNet50,
        'weights': 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'size': (224, 224, 3),
        'port': 9003,
    },
    'xception': {
        'class': Xception,
        'weights': 'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
        'size': (299, 299, 3),
        'port': 9004,
    },
}


def setup_model(name):
    model_path = '.cache/models/%s.h5' % (name, )
    tf_model_dir = '.cache/models/tf/%s' % (name, )

    if not os.path.exists(model_path):
        weights_path = '.cache/weights/%s' % (MODELS[name]['weights'], )
        model_class = MODELS[name]['class']
        model = model_class(weights='imagenet', include_top=False, input_shape=MODELS[name]['size'])
        model.load_weights(weights_path)
        model.save(model_path)

    if os.path.exists(tf_model_dir):
        shutil.rmtree(tf_model_dir)

    return (model_path, tf_model_dir)


def restart_serving_container(model_name):
    time.sleep(0.5)
    call(['docker-compose', 'restart', '%s_serving' % (model_name, )])


def assert_converted_model(tf_model_dir):
    assert os.path.exists(tf_model_dir)
    assert os.path.exists(tf_model_dir + '/variables')
    assert os.path.exists(tf_model_dir + '/variables/variables.data-00000-of-00001')
    assert os.path.exists(tf_model_dir + '/variables/variables.index')
    assert os.path.exists(tf_model_dir + '/saved_model.pb')


def assert_model_serving(model_name):
    attempt = 1
    while True:
        try:
            client = TFClient('localhost', str(MODELS[model_name]['port']))
            img = load_image('tests/fixtures/files/cat.jpg', MODELS[model_name]['size'])
            result = client.make_prediction(img, 'image', timeout=10, name=model_name)
            assert result
            assert 1 == len(result['class_probabilities'])
            # assert 7 == len(result['class_probabilities'][0])
            # assert 7 == len(result['class_probabilities'][0][0])
            # assert 1024 == len(result['class_probabilities'][0][0][0])
            break
        except AbortionError as e:
            if e.details != 'Endpoint read failed' or attempt > 5:
                raise
            time.sleep(1)
            attempt += 1


def load_image(image_path, target_size):
    img = image.load_img(image_path, target_size=target_size[0:2])
    x = image.img_to_array(img)
    # TODO: x = preprocessing(x)
    return np.expand_dims(x, axis=0)


def test_convert_imagenet_inception_v3():
    model_name = 'inception'
    model_path, tf_model_dir = setup_model(model_name)
    KerasToTensorflow.convert(model_path, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name)


def test_convert_imagenet_mobilenet():
    model_name = 'mobilenet'
    model_path, tf_model_dir = setup_model(model_name)
    KerasToTensorflow.convert(model_path, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name)


def test_convert_imagenet_resnet50():
    model_name = 'resnet50'
    model_path, tf_model_dir = setup_model(model_name)
    KerasToTensorflow.convert(model_path, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name)


def test_convert_imagenet_xception():
    model_name = 'xception'
    model_path, tf_model_dir = setup_model(model_name)
    KerasToTensorflow.convert(model_path, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name)
