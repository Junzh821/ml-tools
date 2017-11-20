from __future__ import division

import os
import numpy as np
from ml_tools.compute_dataset_stats import create_class_histogram
from ml_tools import MultiDirectoryIterator

from keras.layers import Dense, Activation
from keras.models import Model
from keras import optimizers
from keras.preprocessing import image
from keras.layers.merge import concatenate
from keras.applications.mobilenet import MobileNet


def make_dir_iterator(path, batch_size):
    return image.ImageDataGenerator().flow_from_directory(path, batch_size=batch_size,
                                                          target_size=(224, 224), seed=0)


# since training is done on multiple threads, without a lock, the iterators in MultiDirectoryIterator get out of sync
# the number of images is not a multiple of batch_size, so therefore the iterators would present blocks not of the same shape which is not allowed
# this test is that a shape mismatch error doesn't get thrown during training (that would indicate the race condition)
def test_multi_directory_iterator_race_condition(sample_dataset_dir):
    n_models = 2
    batch_size = 4
    train_path = os.path.join(sample_dataset_dir, 'Training')
    val_path = os.path.join(sample_dataset_dir, 'Validation')

    # set up training and validation generators
    train_gen = MultiDirectoryIterator([make_dir_iterator(train_path, batch_size) for _ in range(n_models)])
    val_gen = MultiDirectoryIterator([make_dir_iterator(val_path, batch_size) for _ in range(n_models)])

    # join some MobileNets

    base_models = []
    for i in range(n_models):
        model = MobileNet(weights=None)
        for layer in model.layers:
            layer.name += str(i)
        base_models.append(model)

    x = concatenate([m.output for m in base_models])
    x = Dense(create_class_histogram(train_path).shape[0], name='dense')(x)
    x = Activation('softmax', name='act_softmax')(x)

    joined_model = Model([m.input for m in base_models], x)

    # run a few epochs

    joined_model.compile(optimizer=optimizers.SGD(), loss='categorical_crossentropy')

    joined_model.fit_generator(train_gen, validation_data=val_gen, epochs=4, workers=16,
                               steps_per_epoch=int(np.ceil(train_gen.samples / batch_size)),
                               validation_steps=int(np.ceil(val_gen.samples / batch_size)))

    # intentionally no assert, test passes if nothing throws


def test_multi_directory_iterator_block_order(sample_dataset_dir):
    n_iterators = 5
    batch_size = 4
    train_path = os.path.join(sample_dataset_dir, 'Training')

    train_gen = MultiDirectoryIterator([make_dir_iterator(train_path, batch_size) for _ in range(n_iterators)])

    for _ in range(500):
        batches_x, _ = train_gen.next()
        for i in range(1, n_iterators):
            np.testing.assert_array_equal(batches_x[0], batches_x[i])
