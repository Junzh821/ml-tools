import os
import csv
import logging
import sys
import json


def create_folders(path):
    if not os.path.exists(path):
        print 'Creating path: ', path
        os.makedirs(path)
    else:
        print 'Path already exists'


def create_dataset(dataset_json):
    i = 0
    j = 0
    pool_classes = []

    with open(dataset_json) as data_file:
        data = json.load(data_file)

    print data

    csv_file = data['csv_path']
    dataset_path = data['symbolic_dataset_path']
    dataset_name = data['dataset_name']
    subset = data['subset']

    with open(csv_file, 'rb') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            src = row['filename']
            img_name = src.split('/')[-1]
            class_name = row['class_name']

            if class_name not in pool_classes:
                pool_classes.append(class_name)
                create_folders(dataset_path + '/' + dataset_name + '/' + subset + '/' + class_name)

            # Create Symbolic Link
            try:
                dst = dataset_path + '/' + dataset_name + '/' + subset + '/' + class_name + '/' + img_name
                os.symlink(src, dst)
            except:
                # Error
                print('Error processing image: %s', src)
                j += 1

            i += 1

    print ('Total images read: ', i)
    print ('Correct images: ', i-j)
    print ('Error images: ', j)
    sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print('Usage: python create_symbolic_dataset <dataset_json> ')
        print('<dataset_json>: path to json file with dataset info')
        sys.exit(1)

    dataset_json = sys.argv[1]

    create_dataset(dataset_json)