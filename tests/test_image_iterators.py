from __future__ import division

import os
import numpy as np
from ml_tools.compute_dataset_stats import create_class_histogram
from ml_tools import MultiDirectoryIterator

from keras.preprocessing.image import DirectoryIterator
from keras.layers import Dense, Activation
from keras.models import Model
from keras import optimizers
from keras.preprocessing import image
from keras.layers.merge import concatenate
from keras.applications.mobilenet import MobileNet


def test_multi_directory_iterator(sample_dataset_dir):
  batch_size = 4
  train_path = os.path.join(sample_dataset_dir, 'Training')
  val_path = os.path.join(sample_dataset_dir, 'Validation')

  # set up training and validation generators
  def make_dir_iterator(path):
    return image.ImageDataGenerator().flow_from_directory(path, batch_size=batch_size, 
                                                          target_size=(224, 224))
  train_gen = MultiDirectoryIterator([make_dir_iterator(train_path), make_dir_iterator(train_path)])
  val_gen = MultiDirectoryIterator([make_dir_iterator(val_path), make_dir_iterator(val_path)])

  # join 2 MobileNets

  model1 = MobileNet(weights=None)
  for layer in model1.layers:
    layer.name += '_1'

  model2 = MobileNet(weights=None)
  for layer in model2.layers:
    layer.name += '_2'

  x = concatenate([model1.output, model2.output])
  x = Dense(create_class_histogram(train_path).shape[0], name='dense')(x)
  x = Activation('softmax', name='act_softmax')(x)

  model = Model([model1.input, model2.input], x)

  # run a few epochs

  model.compile(optimizer=optimizers.SGD(), loss='categorical_crossentropy')

  model.fit_generator(train_gen, validation_data=val_gen, epochs=4, workers=16,
                      steps_per_epoch=int(np.ceil(train_gen.samples / batch_size)),
                      validation_steps=int(np.ceil(val_gen.samples / batch_size)))