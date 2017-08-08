from ml_tools import get_model_spec
from ml_tools.model_spec import MODEL_SPECS


def assert_valid_model_spec(model_spec_name):
    spec = get_model_spec(model_spec_name)
    assert spec is not None
    assert len(spec.target_size) == 3
    assert spec.klass is not None
    assert callable(spec.preprocess_input)
    image_data = spec.load_image('tests/fixtures/files/cat.jpg')
    assert image_data.any()


def test_get_model_spec():
    for spec_name in MODEL_SPECS.keys():
        assert_valid_model_spec(spec_name)
