#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import csv
from ml_tools.utils import list_files

def create_csv(rootdir):
    first = True
    label = 0
    i = 0
    total_imgs = 0
    partition = os.path.basename(rootdir.strip('/')) 
    dataset_path = rootdir.strip(partition+'/')
    dataset_name = os.path.basename(dataset_path)
    csv_filename = os.path.join(rootdir, dataset_name + '_' + partition + '.csv')

    with open(csv_filename, "w") as csv_file:
        fieldnames = ['filename', 'label', 'class_name']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for dirpath, dirnames, filenames in os.walk(rootdir):
            if first:
                parent_classes = dirnames
                first = False

            if os.path.basename(dirpath) in parent_classes:
                images = list_files(dirpath)
                for image in images:
                        filename = os.path.join(dirpath, image)
                        # Not write '.DS_Store' file as img
                        if '.DS_Store' not in filename:
                            writer.writerow({'filename': filename, 'label': label, 'class_name': os.path.basename(dirpath)})
                            i += 1
                print('Found ' + str(i) + ' images from parent class ' + os.path.basename(dirpath))
                label += 1
                total_imgs += i
                i = 0
    print('CSV file created: ', csv_filename)
    print('Total images: ', total_imgs)
    return csv_filename
