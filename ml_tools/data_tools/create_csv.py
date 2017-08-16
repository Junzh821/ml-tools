#!/usr/bin/env python
import os
import sys
import csv


def create_csv(rootdir):
    first = True
    label = 0
    i = 0
    total_imgs = 0
    partition = os.path.basename(rootdir.strip('/')) 
    dataset_path = rootdir.strip(partition+'/')
    dataset_name = os.path.basename(dataset_path)
    csv_filename = os.path.join(rootdir, dataset_name + '_' + partition + '.csv')

    with open(csv_filename, "wb") as csv_file:
        fieldnames = ['filename', 'label', 'class_name']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for dirpath, dirnames, filenames in os.walk(rootdir):
            if first:
                parent_classes = dirnames
                first = False

            if os.path.basename(dirpath) in sorted(parent_classes):
                for sub_dirpath, sub_dirnames, sub_filenames in os.walk(dirpath):
                    for image in sub_filenames:
                        filename = os.path.join(sub_dirpath, image)
                        # Not write '.DS_Store' file as img
                        if '.DS_Store' not in filename:
                            writer.writerow({'filename': filename, 'label': label, 'class_name': os.path.basename(dirpath)})
                            i += 1
                            total_imgs += 1
                print('Found ' + str(i) + ' images from parent class ' + os.path.basename(dirpath))
                label += 1
                i = 0
    print('CSV file created: ', csv_filename)
    print('Total images: ', total_imgs)
    return csv_filename


if __name__ == "__main__":
    if len(sys.argv) < 1:
        print('Usage: script create_csv <partition_path> ')
        print('<partition_path>: dataset partition directory where class folders are stored')
        sys.exit(1)
    partition_path = sys.argv[1]
    create_csv(partition_path)
