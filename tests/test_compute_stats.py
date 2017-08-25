from ml_tools import compute_dataset_stats as cds
import os
from keras.preprocessing import image
import pytest
import numpy as np


# Test to see if returns the correct number of total images
def test_n_images_total(dataset_path):
    n_images_train = 18
    data_path = os.path.join(dataset_path, 'Training')

    # From Path
    n_total_images = cds.compute_n_images(data_path)
    assert n_images_train == n_total_images, 'N_images from data path fail'

    # From Generator
    train_datagen = image.ImageDataGenerator()

    generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(10, 10),
        batch_size=10,
        shuffle=False,
        class_mode='sparse')
    n_total_images = cds.compute_n_images(data_path, generator)

    assert n_images_train == n_total_images, 'N_images from generator fail'


# Test to see if mean and std have 3 components (BGR) / They are np.float64 values
def test_compute_mean(dataset_path):
    data_path = os.path.join(dataset_path, 'Training')
    n_components = 3

    mean, std = cds.compute_mean_std(data_path, target_size=(100, 100), batch_size=10)

    assert n_components == mean.shape[0], 'N_components mean from data path fail'
    assert n_components == std.shape[0], 'N_components std from data path fail'

    assert isinstance(mean[0], np.float64), "Wrong type!"

    # From Generator
    train_datagen = image.ImageDataGenerator()

    generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(10, 10),
        batch_size=10,
        shuffle=False,
        class_mode='sparse')

    mean, std = cds.compute_mean_std(data_path, target_size=(100, 100), batch_size=10, generator=generator)

    assert n_components == mean.shape[0], 'N_components mean from generator fail'

    assert isinstance(mean[0], np.float64), "Wrong type!"


# Test to check if it is returning the correct number of classes and that the numbers are ints
def test_compute_class_histogram(dataset_path):
    data_path = os.path.join(dataset_path, 'Training')
    n_classes = 2

    class_hist = cds.create_class_histogram(data_path)

    assert n_classes == class_hist.shape[0], 'N_classes from data path fail'

    assert isinstance(class_hist[0], np.int64), "Wrong type!, should be int"

    # From Generator
    train_datagen = image.ImageDataGenerator()

    generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(10, 10),
        batch_size=10,
        shuffle=False,
        class_mode='sparse')

    class_hist = cds.create_class_histogram(data_path, generator)

    assert n_classes == class_hist.shape[0], 'N_classes from generator fail'

    assert isinstance(class_hist[0], np.int64), "Wrong type!, should be int"


# Test to see if mean and std have 3 components (BGR) / They are np.float64 values , class hist and n_total_img work
def test_compute_stats(dataset_path):
    data_path = os.path.join(dataset_path, 'Training')
    n_components = 3
    n_classes = 2
    n_images_train = 18

    dict_stats = cds.compute_stats(data_path, target_size=(100, 100), batch_size=10)

    assert n_components == dict_stats['mean'].shape[0], 'N_components mean from data path fail'
    assert n_components == dict_stats['std'].shape[0], 'N_components std from data path fail'

    assert isinstance(dict_stats['mean'][0], np.float64), "Wrong type!"

    assert n_classes == dict_stats['class_histogram'].shape[0], 'N_classes from generator fail'

    assert isinstance(dict_stats['class_histogram'][0], np.int64), "Wrong type!, should be int"

    assert n_images_train == dict_stats['n_images'], 'N_images from data path fail'

    # From Generator
    train_datagen = image.ImageDataGenerator()

    generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(10, 10),
        batch_size=10,
        shuffle=False,
        class_mode='sparse')

    dict_stats = cds.compute_stats(data_path, target_size=(100, 100), batch_size=10, generator=generator)

    assert n_components == dict_stats['mean'].shape[0], 'N_components mean from generator fail'
    assert n_components == dict_stats['std'].shape[0], 'N_components std from generator fail'

    assert isinstance(dict_stats['mean'][0], np.float64), "Wrong type!"

    assert n_classes == dict_stats['class_histogram'].shape[0], 'N_classes from generator fail'

    assert isinstance(dict_stats['class_histogram'][0], np.int64), "Wrong type!, should be int"

    assert n_images_train == dict_stats['n_images'], 'N_images from generator fail'
