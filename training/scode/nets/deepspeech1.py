import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfl

# ==================================================================================================


class MyModel(tf.keras.Model):  # pylint: disable=abstract-method
    def __init__(self, c_input: int, c_output: int, netconfig: dict):
        super().__init__()

        self.n_hidden = 2048
        self.relu_clip = 20
        self.dropout_rate = 0.05

        self.n_input = c_input
        self.n_output = c_output

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

        kfilter = np.eye(self.window_size)
        kfilter = np.reshape(
            kfilter, [self.window_width, self.n_input, self.window_size]
        )
        eye_filter = tf.constant(kfilter, dtype=dtype)
        return eye_filter

    # ==============================================================================================

    def make_model(self):
        input_tensor = tfl.Input(shape=[None, self.n_input], name="input")

        # Used for easier debugging changes
        x = tf.identity(input_tensor)

        # Create overlapping windows, returns shape [batch_size, steps, window_width * n_input]
        x = tfl.Conv1D(
            self.window_size,
            [self.window_width],
            kernel_initializer=self.window_kernel_init,
            padding="same",
            trainable=False,
        )(x)

        # Dense 1
        x = tfl.TimeDistributed(tfl.Dense(self.n_hidden))(x)
        x = tfl.ReLU(max_value=self.relu_clip)(x)
        x = tfl.Dropout(rate=self.dropout_rate)(x)

        # Dense 2
        x = tfl.TimeDistributed(tfl.Dense(self.n_hidden))(x)
        x = tfl.ReLU(max_value=self.relu_clip)(x)
        x = tfl.Dropout(rate=self.dropout_rate)(x)

        # Dense 3
        x = tfl.TimeDistributed(tfl.Dense(self.n_hidden))(x)
        x = tfl.ReLU(max_value=self.relu_clip)(x)
        x = tfl.Dropout(rate=self.dropout_rate)(x)

        # LSTM 4
        x = tfl.LSTM(self.n_hidden, return_sequences=True, stateful=False)(x)

        # Dense 5
        x = tfl.TimeDistributed(tfl.Dense(self.n_hidden))(x)
        x = tfl.ReLU(max_value=self.relu_clip)(x)
        x = tfl.Dropout(rate=self.dropout_rate)(x)

        # Dense 6
        x = tfl.TimeDistributed(tfl.Dense(self.n_output))(x)

        x = tf.cast(x, dtype="float32")
        x = tf.nn.log_softmax(x)
        output_tensor = tf.identity(x, name="output")

        model = tf.keras.Model(input_tensor, output_tensor, name="DeepSpeech1")
        return model

    # ==============================================================================================

    @staticmethod
    def get_time_reduction_factor():
        """Keep for compatibility with other models"""
        return 1

    # ==============================================================================================

    def summary(self, line_length=100, **kwargs):  # pylint: disable=arguments-differ
        print("")
        self.model.summary(line_length=line_length, **kwargs)

    # ==============================================================================================

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, training=False):  # pylint: disable=arguments-differ
        """Call with input shape: [batch_size, steps_a, n_input].
        Outputs a tensor of shape: [batch_size, steps_b, n_output]"""

        x = self.model(x, training=training)
        return x
