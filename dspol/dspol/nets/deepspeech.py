import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, Input, ReLU

# ==================================================================================================


class DeepSpeech(Model):
    def __init__(self, batch_size):
        super(DeepSpeech, self).__init__()

        self.n_input = 26
        self.n_context = 26
        self.n_hidden = 2048
        self.relu_clip = 20
        self.dropout_rate = 0.05

        alphabet = " abcdefghijklmnopqrstuvwxyz'"
        self.n_output = len(alphabet) + 1

        self.clipped_relu = ReLU(max_value=self.relu_clip)
        self.dense1 = Dense(self.n_hidden)
        self.dense2 = Dense(self.n_hidden)
        self.dense3 = Dense(self.n_hidden)
        # self.lstm = LSTM(self.n_hidden, return_sequences=True, stateful=True)
        self.lstm = LSTM(self.n_hidden, return_sequences=True, stateful=False)
        self.dense5 = Dense(self.n_hidden)
        self.dense6 = Dense(self.n_output)

        # Create a constant convolution filter using an identity matrix, so that the
        # convolution returns patches of the input tensor as is,
        # and we can create overlapping windows over the features.
        self.window_width = 2 * self.n_context + 1
        win_size = self.window_width * self.n_input
        filter = np.eye(win_size)
        filter = filter.reshape(self.window_width, self.n_input, win_size)
        # self.eye_filter = tf.constant(filter, tf.float16)
        self.eye_filter = tf.constant(filter, tf.float32)
        # self.context_window = Conv1D(filters=eye_filter, stride=1, padding='SAME')

    # ==============================================================================================

    def reset_states(self):
        pass
        # self.lstm.reset_states()

    # ==============================================================================================

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, training=False):
        """Call with input shape: [batch_size, n_steps, n_input]"""

        # x = self.feat_in(x)

        batch_size = tf.shape(x)[0]

        # x = tf.cast(x, dtype= tf.float16)

        # Create overlapping windows
        # x = self.context_window(x)
        x = tf.nn.conv1d(x, filters=self.eye_filter, stride=1, padding="SAME")

        # Remove dummy depth dimension and reshape into [batch_size, steps, window_width, n_input]
        x = tf.reshape(x, [batch_size, -1, self.window_width, self.n_input])

        # Permute n_steps and batch_size
        x = tf.transpose(a=x, perm=[1, 0, 2, 3])

        # Transform tensor with shape [batch_size, steps, window_width, n_input] to two dimensions
        # [batch_size * steps, window_width * n_input] which are required for the dense layer
        x = tf.reshape(x, shape=[-1, self.window_width * self.n_input])

        x = self.dense1(x)
        x = self.clipped_relu(x)
        x = Dropout(rate=self.dropout_rate)(x)

        x = self.dense2(x)
        x = self.clipped_relu(x)
        x = Dropout(rate=self.dropout_rate)(x)

        x = self.dense3(x)
        x = self.clipped_relu(x)
        x = Dropout(rate=self.dropout_rate)(x)

        # Reshape tensor as the LSTM expects its input to be
        # of shape [max_time, batch_size, input_size]
        x = tf.reshape(x, [-1, batch_size, self.n_hidden])

        x = self.lstm(x)

        # Reshape again for the dense layer
        x = tf.reshape(x, shape=[-1, self.n_hidden])

        x = self.dense5(x)
        x = self.clipped_relu(x)
        x = Dropout(rate=self.dropout_rate)(x)

        x = self.dense6(x)

        # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
        # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
        # Note, that this differs from the input in that it is time-major.
        x = tf.reshape(x, [-1, batch_size, self.n_output])
        # x = tf.cast(x, dtype=tf.float32)
        return x
