import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

from planktonclas.utils import get_custom_objects

from .saliency import SaliencyMask


class GuidedBackprop(SaliencyMask):
    """A SaliencyMask class that computes saliency masks with GuidedBackProp."""

    GuidedReluRegistered = False

    def __init__(self, model, output_index=0, custom_loss=None):
        """Constructs a GuidedBackprop SaliencyMask."""

        if GuidedBackprop.GuidedReluRegistered is False:

            @tf.RegisterGradient("GuidedRelu")
            def _GuidedReluGrad(op, grad):
                gate_g = tf.cast(grad > 0, "float32")
                gate_y = tf.cast(op.outputs[0] > 0, "float32")
                return gate_y * gate_g * grad

        GuidedBackprop.GuidedReluRegistered = True

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "gb_keras.h5")
            ckpt_prefix = os.path.join(tmpdirname, "guided_backprop_ckpt")

            model.save(model_path)
            with tf.Graph().as_default():
                with tf.Session().as_default():
                    K.set_learning_phase(0)

                    custom_objects = get_custom_objects()
                    custom_objects.update({"custom_loss": custom_loss})

                    load_model(model_path, custom_objects=custom_objects)
                    session = K.get_session()
                    tf.train.export_meta_graph()

                    saver = tf.train.Saver()
                    saver.save(session, ckpt_prefix)

            self.guided_graph = tf.Graph()
            with self.guided_graph.as_default():
                self.guided_sess = tf.Session(graph=self.guided_graph)

                with self.guided_graph.gradient_override_map({"Relu": "GuidedRelu"}):
                    saver = tf.train.import_meta_graph(ckpt_prefix + ".meta")
                    saver.restore(self.guided_sess, ckpt_prefix)

                    self.imported_y = self.guided_graph.get_tensor_by_name(
                        model.output.name
                    )[0][output_index]
                    self.imported_x = self.guided_graph.get_tensor_by_name(
                        model.input.name
                    )

                    self.guided_grads_node = tf.gradients(
                        self.imported_y, self.imported_x
                    )

    def get_mask(self, input_image):
        """Returns a GuidedBackprop mask."""
        x_value = np.expand_dims(input_image, axis=0)
        guided_feed_dict = {}
        guided_feed_dict[self.imported_x] = x_value

        gradients = self.guided_sess.run(
            self.guided_grads_node, feed_dict=guided_feed_dict
        )[0][0]

        return gradients
