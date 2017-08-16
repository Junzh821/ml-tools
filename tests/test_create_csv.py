import pytest
import os
import csv
import tarfile
from backports.tempfile import TemporaryDirectory

from ml_tools.data_tools.create_csv import create_csv


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


def test_create_csv(dataset_path):
    subset = 'Training'
    partition_path = os.path.join(dataset_path, subset)
    csv_path = create_csv(partition_path)
    with open(csv_path, 'rb') as actual, open(os.path.join(dataset_path, 'expected_dataset_Training.csv'), 'rb') as expected:
        actual_dataset = csv.reader(actual, delimiter=',')
        expected_dataset = csv.reader(expected, delimiter=',')
        actual_csv = list(actual_dataset)[1:]
        for actual_file in actual_csv:
            actual_file[0] = actual_file[0].replace(dataset_path, '').strip('/')

        expected_csv = list(expected_dataset)[1:]
        for expected_file in expected_csv:
            expected_file[0] = expected_file[0].replace('./dataset_test/', '')

    assert sorted(actual_csv) == sorted(expected_csv)
