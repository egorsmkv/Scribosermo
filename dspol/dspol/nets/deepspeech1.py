import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers as tfl

# ==================================================================================================


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.n_input = 26
        self.n_hidden = 2048
        self.relu_clip = 20
        self.dropout_rate = 0.05

        alphabet = " abcdefghijklmnopqrstuvwxyz'"
        self.n_output = len(alphabet) + 1

        self.n_context = 9
        self.window_width = 2 * self.n_context + 1
        self.window_size = self.window_width * self.n_input

        self.model = self.make_model()

    # ==============================================================================================

    def window_kernel_init(self, shape, dtype=None):
        """Create a constant convolution filter using an identity matrix, so that the
        convolution returns patches of the input tensor as is, and we can create
        overlapping windows over the features.
        Using extra kernel init function to make the window layer exportable as saved model"""

        filter = np.eye(self.window_size)
        filter = filter.reshape(self.window_width, self.n_input, self.window_size)
        eye_filter = tf.constant(filter, dtype=dtype)
        return eye_filter

    # ==============================================================================================

    def make_model(self):
        """Build sequential model. This runs as fast as autograph conversion."""

        model = tf.keras.Sequential(name="DeepSpeech1")
        model.add(tfl.Input(shape=[None, self.n_input], name="input"))

        # Create overlapping windows, returns shape [batch_size, steps, window_width * n_input]
        window_conv = tfl.Conv1D(
            self.window_size,
            [self.window_width],
            kernel_initializer=self.window_kernel_init,
            padding="SAME",
            trainable=False,
        )
        model.add(window_conv)

        # Dense 1
        model.add(tfl.TimeDistributed(tfl.Dense(self.n_hidden)))
        model.add(tfl.ReLU(max_value=self.relu_clip))
        model.add(tfl.Dropout(rate=self.dropout_rate))

        # Dense 2
        model.add(tfl.TimeDistributed(tfl.Dense(self.n_hidden)))
        model.add(tfl.ReLU(max_value=self.relu_clip))
        model.add(tfl.Dropout(rate=self.dropout_rate))

        # Dense 3
        model.add(tfl.TimeDistributed(tfl.Dense(self.n_hidden)))
        model.add(tfl.ReLU(max_value=self.relu_clip))
        model.add(tfl.Dropout(rate=self.dropout_rate))

        # LSTM 4
        model.add(tfl.LSTM(self.n_hidden, return_sequences=True, stateful=False))

        # Dense 5
        model.add(tfl.TimeDistributed(tfl.Dense(self.n_hidden)))
        model.add(tfl.ReLU(max_value=self.relu_clip))
        model.add(tfl.Dropout(rate=self.dropout_rate))

        # Dense 6
        model.add(tfl.TimeDistributed(tfl.Dense(self.n_output)))

        # Transpose to shape [n_steps, batch_size, n_output]
        model.add(tfl.Lambda(lambda x: tf.transpose(x, perm=[1, 0, 2])))

        # Named output layer
        model.add(tfl.Lambda(lambda x: x, name="output"))

        return model

    # ==============================================================================================

    def reset_states(self):
        # self.lstm.reset_states()
        pass

    # ==============================================================================================

    def call(self, x, training=False):
        """Call with input shape: [batch_size, n_steps, n_input]"""

        x = self.model(x)
        return x
