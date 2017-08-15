import pytest
import os
import tarfile
import json
from backports.tempfile import TemporaryDirectory

from ml_tools.data_tools.create_csv import create_csv
from ml_tools.data_tools.create_symbolic_dataset import create_symbolic_dataset


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


def test_create_symbolic_datset(dataset_path):

    subset = 'Training'
    csv_path = create_csv(dataset_path, subset)

    dataset_json = {}
    dataset_json['csv_path'] = csv_path
    dataset_json['symbolic_dataset_path'] = dataset_path
    dataset_json['dataset_name'] = 'dataset_test'
    dataset_json['subset'] = subset

    json_file = os.path.join(dataset_path, 'json_data.json')
    with open(json_file, 'w') as output_file:
        json.dump(dataset_json, output_file)

    create_symbolic_dataset(json_file)
