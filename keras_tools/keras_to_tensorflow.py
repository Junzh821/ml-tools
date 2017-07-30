import os
import keras
import tensorflow


class KerasToTensorflow(object):

    @classmethod
    def convert(cls, model_path, output_dir):
        keras.backend.clear_session()
        session = tensorflow.Session()
        keras.backend.set_session(session)

        # Disable loading of learning nodes
        keras.backend.set_learning_phase(0)

        model = keras.models.load_model(model_path, custom_objects={
            # for mobilenet import, doesn't affect other model types
            'relu6': keras.applications.mobilenet.relu6,
            'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D
        })

        builder = tensorflow.saved_model.builder.SavedModelBuilder(output_dir)
        signature = tensorflow.saved_model.signature_def_utils.predict_signature_def(
            inputs={
                'image': model.layers[0].input
            },
            outputs={
                'class_probabilities': model.layers[-1].output
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
