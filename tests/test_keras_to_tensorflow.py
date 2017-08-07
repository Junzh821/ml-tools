import pytest
import os
import time
import shutil

from subprocess import call

from tensorflow_serving_client import TensorflowServingClient
from grpc.framework.interfaces.face.face import AbortionError

from ml_tools.keras_to_tensorflow import KerasToTensorflow
from ml_tools.utils import MODEL_SPECS, load_image


MODEL_SERVING_PORTS = {
    'inception_v3': 9001,
    'mobilenet_v1': 9002,
    'resnet50': 9003,
    'xception': 9004,
    'vgg16': 9005,
    'vgg19': 9006,
}


def setup_model(name, model_path):
    tf_model_dir = '.cache/models/%s' % (name, )

    model_class = MODEL_SPECS[name]['class']
    model = model_class(weights='imagenet', input_shape=MODEL_SPECS[name]['target_size'])
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    if os.path.exists(tf_model_dir):
        shutil.rmtree(tf_model_dir)
    os.makedirs(os.path.dirname(tf_model_dir), exist_ok=True)

    return tf_model_dir


def restart_serving_container(model_name):
    time.sleep(1)
    call(['docker-compose', 'restart', model_name])


def assert_converted_model(tf_model_dir):
    assert os.path.exists(tf_model_dir)
    assert os.path.exists(tf_model_dir + '/variables')
    assert os.path.exists(tf_model_dir + '/variables/variables.data-00000-of-00001')
    assert os.path.exists(tf_model_dir + '/variables/variables.index')
    assert os.path.exists(tf_model_dir + '/saved_model.pb')


def assert_model_serving(model_name, imagenet_dictionary):
    model_spec = MODEL_SPECS[model_name]
    attempt = 1
    while True:
        try:
            client = TensorflowServingClient('localhost', MODEL_SERVING_PORTS[model_name])
            image_data = load_image('tests/fixtures/files/cat.jpg', model_spec['target_size'])
            response = client.make_prediction(image_data, 'image')
            assert response is not None
            assert len(response['class_probabilities']) == 1
            assert len(response['class_probabilities'][0]) == 1000
            predictions = dict(zip(imagenet_dictionary, response['class_probabilities'][0]))
            top_5 = sorted(predictions.items(), reverse=True, key=lambda kv: kv[1])[:5]
            expected = [
                ('spatula', 0.9628159403800964),
                ('gondola', 0.02030189521610737),
                ('toyshop', 0.003980898763984442),
                ('tray', 0.00305487890727818),
                ('lakeside, lakeshore', 0.001700777793303132)
            ]
            assert top_5 == expected
            break
        except AbortionError as e:
            if e.details != 'Endpoint read failed' or attempt > 5:
                raise
            time.sleep(1)
            attempt += 1


def test_convert_imagenet_inception_v3(temp_file, imagenet_dictionary):
    model_name = 'inception'
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary)


def test_convert_imagenet_mobilenet(temp_file, imagenet_dictionary):
    model_name = 'mobilenet'
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary)


def test_convert_imagenet_resnet50(temp_file, imagenet_dictionary):
    model_name = 'resnet50'
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary)


def test_convert_imagenet_xception(temp_file, imagenet_dictionary):
    model_name = 'xception'
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary)


def test_convert_imagenet_vgg16(temp_file, imagenet_dictionary):
    model_name = 'vgg16'
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary)


def test_convert_imagenet_vgg19(temp_file, imagenet_dictionary):
    model_name = 'vgg19'
    tf_model_dir = setup_model(model_name, temp_file)
    KerasToTensorflow.convert(temp_file, tf_model_dir)
    assert_converted_model(tf_model_dir)
    restart_serving_container(model_name)
    assert_model_serving(model_name, imagenet_dictionary)
