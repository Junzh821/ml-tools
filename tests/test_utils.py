from ml_tools import load_image, get_model_spec
from ml_tools.model_spec import MODEL_SPECS 


def test_load_image():
    for spec_name in MODEL_SPECS.keys():
        model_spec = get_model_spec(spec_name)
        image_data = load_image('tests/fixtures/files/cat.jpg',
                                 model_spec.target_size,
                                 preprocess_input=model_spec.preprocess_input)
        assert image_data.any()
