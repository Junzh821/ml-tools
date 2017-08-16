import numpy as np
from keras.preprocessing import image
import collections
import pickle
import time
import sys
# Solve High Resolution truncated files
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def compute_mean_std(data_path, batch_size, target_size, generator=None):
    if generator is None:
        # Keras img loaders
        train_datagen = image.ImageDataGenerator()

        generator = train_datagen.flow_from_directory(
            data_path,
            target_size=target_size,
            batch_size=batch_size,
            shuffle=False,
            class_mode='sparse')

    mean_aux = np.zeros(3)
    std_aux = np.zeros(3)
    mean = np.zeros(3)
    std = np.zeros(3)
    count_batch = 1
    mean_count = 0

    n_total_images = generator.samples
    n_batches = n_total_images // batch_size
    last_imgs = n_total_images % batch_size

    print('Computing mean and std...')
    print('Total Images', n_total_images)
    print('Using %d Batches of size %d ' % (n_batches, batch_size))
    print('Images in last batch ', last_imgs)
    t_t = time.time()
    # High Res. images takes longer
    for X_batch, y_batch in generator:
        # Last Batch
        t = time.time()
        if count_batch == n_batches + 1:
            for img in X_batch[0:last_imgs]:
                mean_count += 1
                for ch in range(0, 3):
                    mean_aux[ch] += np.mean(img[:, :, ch])
                    std_aux[ch] += np.std(img[:, :, ch])
            mean_aux /= n_total_images
            std_aux /= n_total_images

        elif count_batch > n_batches:
            print('End')
            break

        else:
            for img in X_batch:
                mean_count += 1
                for ch in range(0, 3):
                    mean_aux[ch] += np.mean(img[:, :, ch])
                    std_aux[ch] += np.std(img[:, :, ch])
            mean_aux /= n_total_images
            std_aux /= n_total_images

        print('Batch number ', count_batch)
        print('Time elapsed', time.time() - t)
        sys.stdout.flush()
        mean += mean_aux
        std += std_aux
        count_batch += 1

    print('Mean ', mean)
    print('Std ', std)
    print('Total time elapsed ', time.time() - t_t)
    return mean, std


def create_class_histogram(data_path, generator=None):
    if generator is None:
        # Keras img loaders
        train_datagen = image.ImageDataGenerator()
        # This function would never use them
        generator = train_datagen.flow_from_directory(
            data_path,
            target_size=(10, 10),
            batch_size=10,
            shuffle=False,
            class_mode='sparse')

    # Create histogram of classes
    class_counter = collections.Counter(generator.classes)
    class_histogram = list()
    for k in class_counter.keys():
        print('Class %d has %d images' % (k, class_counter[k]))
        class_histogram.append(class_counter[k])

    # Convert to numpy array
    class_histogram = np.array(class_histogram)

    return class_histogram


def compute_n_images(data_path, generator=None):
    if generator is None:
        # Keras img loaders
        train_datagen = image.ImageDataGenerator()
        # This function would never use them
        generator = train_datagen.flow_from_directory(
            data_path,
            target_size=(10, 10),
            batch_size=10,
            shuffle=False,
            class_mode='sparse')

    print('Number of images in dataset is ', generator.samples)
    return generator.samples


def compute_stats(data_path, target_size=(224, 224), batch_size=500, generator=None, save_name=''):
    '''
    Compute N_images, Mean, Std, Class Histogram from the dataset introduced with the paths/ generator.
    Save info in pickle file if you introduce a save_name.
    I've checked that there is no image repetition per batch. In the end all the images are used to compute the stats.

   :param path_data: path to folder with classes of the dataset.
   :param batch_size: size of batches to process.
   :param target_size: size of the target images.
   :param generator: Keras image generator.
   :param save_name: To save a pickle object with the dictionary
   :return: A dictionary containing {'n_images': n_total_images, 'mean': mean, 'std': std,  'class_histogram': class_histogram}
    '''

    if generator is None:
        # Keras img loaders
        train_datagen = image.ImageDataGenerator()

        generator = train_datagen.flow_from_directory(
            data_path,
            target_size=target_size,
            batch_size=batch_size,
            shuffle=False,
            class_mode='sparse')

    mean, std = compute_mean_std(data_path, batch_size, target_size, generator)
    n_total_images = generator.samples
    class_histogram = create_class_histogram(data_path, generator)

    dict_stats = {'n_images': n_total_images, 'mean': mean, 'std': std, 'class_histogram': class_histogram}

    if save_name != '':
        # Save Stats in pickle file
        file_pickle = save_name
        with open(file_pickle, 'wb') as handle:
            pickle.dump(dict_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Saved pickle file in, ', save_name)

    return dict_stats
