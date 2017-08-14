import pytest
import os
import sys
import csv
import tarfile
import tempfile

import sys
sys.path.append('../')

from ml_tools.data_tools.create_csv import create_csv

from ml_tools.utils import list_files


@pytest.fixture(scope='function')
def haystack_dir():
    tar = tarfile.open('tests/fixtures/files/dataset_test.tar.gz', "r:gz")
    with tempfile.TemporaryDirectory() as temp_dir:
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                output_path = os.path.join(temp_dir, member.name)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'wb') as output_file:
                    output_file.write(f.read())
        yield os.path.join(temp_dir, 'dataset_test')


def test_create_csv(dataset_path, subset):

	actual_dataset = create_csv(dataset_path, subset)
	with  open(actual_dataset, 'rb') as actual, open('tests/fixtures/files/expected_dataset_Training.csv', 'rb') as expected:
		actual_dataset = csv.reader(actual, delimiter=',')
		expected_dataset = csv.reader(expected, delimiter=',')
	
	assert actual_dataset == expected_dataset













