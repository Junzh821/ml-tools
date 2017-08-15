#!/usr/bin/env python
import os
import sys
import csv


def create_csv(rootdir, set):
    first = True
    label = 0
    i = 0
    total_imgs = 0
    file = os.path.join(rootdir, os.path.basename(rootdir) + '_' + set + '.csv')

    with open(file, "wb") as csv_file:
        fieldnames = ['filename', 'label', 'class_name']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for dirpath, dirnames, filenames in os.walk(os.path.join(rootdir, set)):
            if first:
                parent_classes = sorted(dirnames)
                first = False

            if os.path.basename(dirpath) in parent_classes:
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
    print('CSV file created: ', file)
    print('Total images: ', total_imgs)
    return file


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: script create_csv <rootdir> <set>')
        print('<rootdir>: dataset directory where sets are stored')
        print('<set>: string indicating the subset name. i.e. "Training", "Validation" ')
        sys.exit(1)
    dataset_path = sys.argv[1]
    subset = sys.argv[2]
    create_csv(dataset_path, subset)
