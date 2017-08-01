import keras
import tensorflow


class KerasToTensorflow(object):

    @staticmethod
    def load_keras_model(model_path):
        return keras.models.load_model(model_path, custom_objects={
            # for mobilenet import, doesn't affect other model types
            'relu6': keras.applications.mobilenet.relu6,
            'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D
        })

    @staticmethod
    def convert(model_path, output_dir, output_stripped_model_path=None):
        # cut out to_multi_gpu stuff (this could possibly break some models which don't use to_multi_gpu)
        model = KerasToTensorflow.load_keras_model(model_path)
        stripped_model = next((l for l in model.layers if isinstance(l, keras.engine.training.Model)), None)
        if stripped_model:
            if output_stripped_model_path is None:
                output_stripped_model_path = model_path + '-stripped.hdf5'
            stripped_model.save(output_stripped_model_path)
            model_path = output_stripped_model_path

        keras.backend.clear_session()
        session = tensorflow.Session()
        keras.backend.set_session(session)

        # disable loading of learning nodes
        keras.backend.set_learning_phase(0)

        model = KerasToTensorflow.load_keras_model(model_path)

        builder = tensorflow.saved_model.builder.SavedModelBuilder(output_dir)
        signature = tensorflow.saved_model.signature_def_utils.predict_signature_def(
            inputs={
                'image': model.input
            },
            outputs={
                'class_probabilities': model.output
            }
        )

        builder.add_meta_graph_and_variables(
            sess=keras.backend.get_session(),
            tags=[tensorflow.saved_model.tag_constants.SERVING],
            signature_def_map={
                tensorflow.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            }
        )
        builder.save()


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2 or not (sys.argv[1].endswith('.h5') or sys.argv[1].endswith('.hdf5')):
        print('usage: keras_to_tensorflow.py <hdf5_model_file>')
        sys.exit(1)
    KerasToTensorflow.convert(sys.argv[1])
