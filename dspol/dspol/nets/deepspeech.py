import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# ==================================================================================================


class DeepSpeech(Model):
    def __init__(self, batch_size):
        super(DeepSpeech, self).__init__()

        self.batch_size = batch_size
        self.n_input = 26
        self.n_hidden = 2048
        self.relu_clip = 20
        self.dropout_rate = 0.05

        alphabet = " abcdefghijklmnopqrstuvwxyz'"
        self.n_output = len(alphabet) + 1

        # self.feat_in = Input(batch_size=self.batch_size, shape=[None, self.n_input])
        self.dense1 = Dense(self.n_hidden, activation="relu")
        self.dense2 = Dense(self.n_hidden, activation="relu")
        self.dense3 = Dense(self.n_hidden, activation="relu")
        # self.lstm = LSTM(self.n_hidden, return_sequences=True, stateful=True)
        self.lstm = LSTM(self.n_hidden, return_sequences=True, stateful=False)
        self.dense5 = Dense(self.n_hidden, activation="relu")
        self.dense6 = Dense(self.n_output)

    # ==============================================================================================

    def reset_states(self):
        pass
        # self.lstm.reset_states()

    # ==============================================================================================

    @tf.function
    def call(self, x, training=False):
        # x = self.feat_in(x)

        # Transform tensor with shape [batch_size, steps, n_input]
        # to two dimensions [batch_size * steps, n_input] which are required from the dense layer
        x = tf.reshape(x, shape=[-1, self.n_input])

        x = self.dense1(x)
        x = tf.minimum(x, self.relu_clip)
        x = Dropout(rate=self.dropout_rate)(x)

        x = self.dense2(x)
        x = tf.minimum(x, self.relu_clip)
        x = Dropout(rate=self.dropout_rate)(x)

        x = self.dense3(x)
        x = tf.minimum(x, self.relu_clip)
        x = Dropout(rate=self.dropout_rate)(x)

        # Reshape tensor as the LSTM expects its input to be
        # of shape [max_time, batch_size, input_size]
        x = tf.reshape(x, [-1, self.batch_size, self.n_hidden])

        x = self.lstm(x)

        # Reshape again for the dense layer
        x = tf.reshape(x, shape=[-1, self.n_hidden])

        x = self.dense5(x)
        x = tf.minimum(x, self.relu_clip)
        x = Dropout(rate=self.dropout_rate)(x)

        x = self.dense6(x)

        # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_hidden_6]
        # to the slightly more useful shape [n_steps, batch_size, n_hidden_6].
        # Note, that this differs from the input in that it is time-major.
        x = tf.reshape(x, [-1, self.batch_size, self.n_output])

        return x
