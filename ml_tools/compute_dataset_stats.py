import numpy as np
from keras.preprocessing import image
import collections
import pickle
import time
import sys
import argparse
# Solve High Resolution truncated files
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


'''
Compute N_images, Mean, Std, Class Histogram from the dataset introduced with the paths. Save info in pickle file.
It needs a dataset with N class folders and all the images of the class inside the folder.
I've checked that there is no image repetition per batch. In the end all the images are used to compute the stats.
'''

# Dataset paths
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', help='Dataset path', default='/data/images/symbolic_datasets/')
parser.add_argument('--name_dataset', help='Name dataset', default='')
parser.add_argument('--subset', help='(train, val, test)', default='train')

opts = parser.parse_args()

complete_data_path = opts.data_path + '/' + opts.name_dataset + '/' + opts.subset

# Target Size and Batch Size
target_size = (224, 224)
batch_size = 500

# Keras img loaders
train_datagen = image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    complete_data_path,
    target_size=target_size,
    batch_size=batch_size,
    shuffle=False,
    class_mode='sparse')

mean = np.zeros((3, 1))
std = np.zeros((3, 1))
count_batch = 1
mean_count = 0

n_total_images = train_generator.samples
n_batches = n_total_images // batch_size
last_imgs = n_total_images % batch_size

print('Total Images', n_total_images)
print('Using %d Batches of size %d ' % (n_batches, batch_size))
print('Images in last batch ', last_imgs)


# High Res. images takes longer
for X_batch, y_batch in train_generator:
    # Last Batch
    t = time.time()
    if count_batch == n_batches:
        for img in X_batch[0:last_imgs]:
            mean_count += 1
            for ch in range(0, 3):
                mean[ch] += np.mean(img[:, :, ch])
                std[ch] += np.std(img[:, :, ch])

    elif count_batch > n_batches:
        print('END')
        break

    else:
        for img in X_batch:
            mean_count += 1
            for ch in range(0, 3):
                mean[ch] += np.mean(img[:, :, ch])
                std[ch] += np.std(img[:, :, ch])

    print('Batch number ', count_batch)
    print('time batch', time.time() - t)
    sys.stdout.flush()
    mean /= mean_count
    std /= mean_count
    count_batch += 1
    mean_count = 1

# Create histogram of classes
class_counter = collections.Counter(train_generator.classes)
class_histogram = list()

print('||*||-- Stats Dataset --||*||')
print('Total Images ', n_total_images)
for k in class_counter.keys():
    print('Class %d has %d images' % (k, class_counter[k]))
    class_histogram.append(class_counter[k])
print('Mean', mean)
print('Std', std)

# Convert to numpy array
class_histogram = np.array(class_histogram)

# Save Stats in pickle file
file_pickle = opts.data_path + '/' + opts.name_dataset + '/' + opts.name_dataset + '_' + opts.subset + '_stats.pickle'

dictionary = {'n_images': n_total_images, 'mean': mean, 'std': std, 'class_histogram': class_histogram}

with open(file_pickle, 'wb') as handle:
    pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
