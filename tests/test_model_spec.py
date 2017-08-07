from ml_tools import get_model_spec


def test_get_model_spec():
    spec = get_model_spec('mobilenet_v1')
    assert spec is not None
    assert spec.target_size == (224, 224, 3)
    assert spec.klass is not None
    assert callable(spec.preprocess_input)

    spec = get_model_spec('inception_v3')
    assert spec is not None
    assert spec.target_size == (299, 299, 3)
    assert spec.klass is not None
    assert callable(spec.preprocess_input)

    spec = get_model_spec('xception')
    assert spec is not None
    assert spec.target_size == (299, 299, 3)
    assert spec.klass is not None
    assert callable(spec.preprocess_input)

    spec = get_model_spec('resnet50')
    assert spec is not None
    assert spec.target_size == (224, 224, 3)
    assert spec.klass is not None
    assert callable(spec.preprocess_input)

    spec = get_model_spec('vgg16')
    assert spec is not None
    assert spec.target_size == (224, 224, 3)
    assert spec.klass is not None
    assert callable(spec.preprocess_input)

    spec = get_model_spec('vgg19')
    assert spec is not None
    assert spec.target_size == (224, 224, 3)
    assert spec.klass is not None
    assert callable(spec.preprocess_input)
