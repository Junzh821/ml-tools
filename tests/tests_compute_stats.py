from ml_tools import compute_dataset_stats as cds
import os
from keras.preprocessing import image
import pytest
import tarfile
from backports.tempfile import TemporaryDirectory
import numpy as np


@pytest.fixture(scope='function')
def dataset_path():
    tar = tarfile.open('tests/fixtures/files/dataset_test.tar.gz', "r:gz")
    with TemporaryDirectory() as temp_dir:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                output_path = os.path.join(temp_dir, member.name)
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))

                with open(output_path, 'wb') as output_file:
                    output_file.write(f.read())
        yield temp_dir


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
    assert n_components == std.shape[0], 'N_components mean from data path fail'

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
