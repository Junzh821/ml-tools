import pytest
import csv
import codecs
import requests

from tempfile import NamedTemporaryFile
from backports.tempfile import TemporaryDirectory


@pytest.fixture(scope='function')
def temp_dir():
    with TemporaryDirectory() as d:
        yield d


@pytest.fixture(scope='function')
def temp_file():
    with NamedTemporaryFile() as f:
        yield f.name


@pytest.fixture
def imagenet_dictionary():
    response = requests.get('https://storage.googleapis.com/tf-serving-docker-http-eae9e0c7-661d-4cca-836a-0433a8da44ba/imagenet/dictionary.csv')
    reader = csv.reader(response.text.decode('utf-8').splitlines())
    dictionary = dict(list(reader))
    return [dictionary[key] for key in sorted(dictionary)]
