import os

from ml_tools import load_image, get_model_spec, list_files
from ml_tools.model_spec import MODEL_SPECS


def touch(path):
    with open(path, 'a'):
        os.utime(path, None)


def test_load_image():
    for spec_name in MODEL_SPECS.keys():
        model_spec = get_model_spec(spec_name)
        image_data = load_image('tests/fixtures/files/cat.jpg',
                                model_spec.target_size,
                                preprocess_input=model_spec.preprocess_input)
        assert image_data.any()


def test_list_files(temp_dir):
    touch(os.path.join(temp_dir, 'foo.jpg'))
    os.makedirs(os.path.join(temp_dir, 'bar'))
    touch(os.path.join(temp_dir, 'bar', 'baz-1.jpg'))
    touch(os.path.join(temp_dir, 'bar', 'baz-2.jpg'))
    actual = list_files(temp_dir)
    actual = [file.replace(temp_dir + '/', '') for file in actual]
    expected = ['foo.jpg', 'bar/baz-1.jpg', 'bar/baz-2.jpg']
    assert actual == expected


def test_list_files_relative(temp_dir):
    touch(os.path.join(temp_dir, 'foo.jpg'))
    os.makedirs(os.path.join(temp_dir, 'bar'))
    touch(os.path.join(temp_dir, 'bar', 'baz-1.jpg'))
    touch(os.path.join(temp_dir, 'bar', 'baz-2.jpg'))
    actual = list_files(temp_dir, relative=True)
    expected = ['foo.jpg', 'bar/baz-1.jpg', 'bar/baz-2.jpg']
    assert actual == expected
