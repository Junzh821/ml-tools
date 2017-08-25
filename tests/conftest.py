import pytest

from tempfile import NamedTemporaryFile
from backports.tempfile import TemporaryDirectory
import tarfile
import os


@pytest.fixture(scope='function')
def temp_dir():
    with TemporaryDirectory() as d:
        yield d


@pytest.fixture(scope='function')
def temp_file():
    with NamedTemporaryFile() as f:
        yield f.name


@pytest.fixture(scope='session')
def sample_dataset_dir():
    tar = tarfile.open('tests/files/dataset_test.tar.gz', "r:gz")
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
