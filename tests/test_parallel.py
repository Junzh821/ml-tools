import pytest
import numpy as np

from keras.layers import Dense, Dropout
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.callbacks import LambdaCallback
from tensorflow.python.framework.errors import InvalidArgumentError

from ml_tools import make_parallel


def generate_compatible_data(batch_size):
    while True:
        yield ([np.random.random((batch_size, 3)), np.random.random((batch_size, 3))],
               [np.random.random((batch_size, 4)), np.random.random((batch_size, 3))])


def generate_incompatible_data(batch_size):
    while True:
        yield ([np.random.random((batch_size + 1, 3)), np.random.random((batch_size + 1, 3))],
               [np.random.random((batch_size + 1, 4)), np.random.random((batch_size + 1, 3))])


def run_parallel_test(data_generator):
    a = Input(shape=(3,), name='input_a')
    b = Input(shape=(3,), name='input_b')
    a_2 = Dense(4, name='dense_1')(a)
    dp = Dropout(0.5, name='dropout')
    b_2 = dp(b)
    optimizer = 'rmsprop'
    loss = 'mse'
    loss_weights = [1., 0.5]
    model = Model([a, b], [a_2, b_2])
    model = make_parallel(model, 2)
    model.compile(optimizer, loss,
                  metrics=[],
                  loss_weights=loss_weights,
                  sample_weight_mode=None)

    trained_epochs = []
    tracker_cb = LambdaCallback(on_epoch_begin=lambda epoch, logs: trained_epochs.append(epoch))
    out = model.fit_generator(data_generator(4),
                              steps_per_epoch=3,
                              epochs=5,
                              initial_epoch=2,
                              callbacks=[tracker_cb])
    assert trained_epochs == [2, 3, 4]


def test_make_parallel():
    run_parallel_test(generate_compatible_data)


def test_make_parallel_with_incompatible_data():
    run_parallel_test(generate_incompatible_data)
