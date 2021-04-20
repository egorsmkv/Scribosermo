import tensorflow as tf
from tensorflow.keras import layers as tfl

# ==================================================================================================


class MyModel(tf.keras.Model):  # pylint: disable=abstract-method
    def __init__(self, c_input: int, c_output: int, netconfig: dict):
        super().__init__()

        self.n_input = c_input
        self.n_output = c_output

        self.n_lstms = 5
        self.n_hidden = 1024
        self.relu_clip = 20
        self.dropout_rate = 0.05
        self.feature_time_reduction_factor = 2

        self.model = self.make_model()

    # ==============================================================================================

    def make_model(self):
        input_tensor = tfl.Input(shape=[None, self.n_input], name="input")

        # Used for easier debugging changes
        x = tf.identity(input_tensor)

        # Paper did recommend 2D convolutions here, with frequency as first dimension,
        # but this was easier to implement for now
        x = tfl.Conv1D(filters=512, kernel_size=5, padding="same")(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.ReLU(max_value=self.relu_clip)(x)

        x = tfl.Conv1D(filters=512, kernel_size=5, padding="same")(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.ReLU(max_value=self.relu_clip)(x)

        x = tfl.Conv1D(filters=512, kernel_size=5, padding="same", strides=2)(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.ReLU(max_value=self.relu_clip)(x)

        # Paper uses 7 bidirectional LSTMs here. Using unidirectional like in Mozilla's DS1 project
        # would result in about 2x speed up of the training
        for _ in range(self.n_lstms - 1):
            x = tfl.Bidirectional(
                tfl.LSTM(self.n_hidden, return_sequences=True, stateful=False)
            )(x)

            x = tfl.BatchNormalization()(x)
            x = tfl.ReLU(max_value=self.relu_clip)(x)

        x = tfl.Bidirectional(
            tfl.LSTM(self.n_hidden, return_sequences=True, stateful=False)
        )(x)

        x = tfl.TimeDistributed(tfl.Dense(self.n_hidden * 2))(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.ReLU(max_value=self.relu_clip)(x)

        # Predict propabilities over the alphabet
        x = tfl.TimeDistributed(tfl.Dense(self.n_output))(x)

        x = tf.cast(x, dtype="float32")
        x = tf.nn.log_softmax(x)
        output_tensor = tf.identity(x, name="output")

        model = tf.keras.Model(input_tensor, output_tensor, name="DeepSpeech2")
        return model

    # ==============================================================================================

    def get_time_reduction_factor(self):
        """Some models reduce the time dimension of the features, for example with striding.
        When the inputs are padded for better batching, it's complicated to get the original length
        from the outputs. So we use this fixed factor."""
        return self.feature_time_reduction_factor

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
