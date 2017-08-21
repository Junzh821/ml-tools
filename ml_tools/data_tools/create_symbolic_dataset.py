#!/usr/bin/env python
from __future__ import print_function
import os
import csv
import sys
import json


def create_folders(path):
    if not os.path.exists(path):
        print ('Creating path: ', path)
        os.makedirs(path)
    else:
        print ('Path already exists')


def create_symbolic_dataset(dataset_json):
    total_images = 0
    failed_images = 0
    pool_classes = []

    with open(dataset_json, 'r') as data_file:
        data = json.load(data_file)

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
                create_folders(os.path.join(dataset_path, dataset_name, subset, class_name))

            # Create Symbolic Link
            try:
                dst = os.path.join(dataset_path, dataset_name, subset, class_name, img_name)
                os.symlink(src, dst)
            except BaseException:
                # Error
                print('Error processing image: %s', src)
                failed_images += 1

            total_images += 1

    print ('Total images read: ', total_images)
    print ('Correct images: ', total_images - failed_images)
    print ('Error images: ', failed_images)
    print ('Symbolic dataset created in ', dataset_path)
    sys.stdout.flush()
