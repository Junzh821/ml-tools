import os
import keras
import numpy as np
import tensorflow as tf


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
                output_stripped_model_path = '%s-stripped%s' % os.path.splitext(model_path)
            stripped_model.save(output_stripped_model_path)
            model_path = output_stripped_model_path

        keras.backend.clear_session()
        session = tf.Session()
        keras.backend.set_session(session)

        # disable loading of learning nodes
        keras.backend.set_learning_phase(0)

        model = KerasToTensorflow.load_keras_model(model_path)

        builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
        signature = tf.saved_model.signature_def_utils.predict_signature_def(
            inputs={'images': tf.convert_to_tensor(model.inputs)},
            outputs={'predictions': tf.convert_to_tensor(model.outputs)}
        )

        builder.add_meta_graph_and_variables(
            sess=keras.backend.get_session(),
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            }
        )
        builder.save()


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3 or not (sys.argv[1].endswith('.h5') or sys.argv[1].endswith('.hdf5')):
        print('Usage: keras_to_tf.py <hdf5_model_file> <output_dir>')
        sys.exit(1)
    KerasToTensorflow.convert(sys.argv[1], sys.argv[2])
