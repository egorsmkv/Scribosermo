import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers as tfl

# ==================================================================================================


class MyModel(Model):  # pylint: disable=abstract-method
    def __init__(self, c_input, c_output):
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
        """Build sequential model. This did run faster than autograph conversion."""

        model = tf.keras.Sequential(name="DeepSpeech2")
        model.add(tfl.Input(shape=[None, self.n_input], name="input"))

        # Paper did recommend 2D convolutions here, with frequency as first dimension,
        # but this was easier to implement for now
        model.add(tfl.Conv1D(filters=512, kernel_size=5))
        model.add(tfl.BatchNormalization())
        model.add(tfl.ReLU(max_value=self.relu_clip))

        model.add(tfl.Conv1D(filters=512, kernel_size=5))
        model.add(tfl.BatchNormalization())
        model.add(tfl.ReLU(max_value=self.relu_clip))

        model.add(tfl.Conv1D(filters=512, kernel_size=5, strides=2))
        model.add(tfl.BatchNormalization())
        model.add(tfl.ReLU(max_value=self.relu_clip))

        # Paper uses 7 bidirectional LSTMs here. Using unidirectional like in Mozilla's DS1 project
        # would result in about 2x speed up of the training
        for _ in range(self.n_lstms - 1):
            model.add(
                tfl.Bidirectional(
                    tfl.LSTM(self.n_hidden, return_sequences=True, stateful=False)
                )
            )
            model.add(tfl.BatchNormalization())
            model.add(tfl.ReLU(max_value=self.relu_clip))
        model.add(
            tfl.Bidirectional(
                tfl.LSTM(self.n_hidden, return_sequences=True, stateful=False)
            )
        )

        model.add(tfl.TimeDistributed(tfl.Dense(self.n_hidden * 2)))
        model.add(tfl.BatchNormalization())
        model.add(tfl.ReLU(max_value=self.relu_clip))

        # Predict propabilities over our alphabet
        model.add(tfl.TimeDistributed(tfl.Dense(self.n_output)))

        model.add(tfl.Lambda(lambda x: x, name="output"))
        return model

    # ==============================================================================================

    # Input signature is required to export this method into ".pb" format and use it while testing
    @tf.function(input_signature=[])
    def get_time_reduction_factor(self):
        """Some models reduce the time dimension of the features, for example with striding.
        When the inputs are padded for better batching, it's complicated to get the original length
        from the outputs. So we use this fixed factor."""
        return self.feature_time_reduction_factor

    # ==============================================================================================

    def summary(self, line_length=100, **kwargs):  # pylint: disable=arguments-differ
        self.model.summary(line_length=line_length, **kwargs)

    # ==============================================================================================

    # This input signature is required that we can export and load the model in ".pb" format
    # with a variable sequence length, instead of using the one of the first input.
    @tf.function(input_signature=[tf.TensorSpec([None, None, None], tf.float32)])
    def call(self, x):  # pylint: disable=arguments-differ
        """Call with input shape: [batch_size, steps_a, n_input],
        outputs tensor of shape: [batch_size, steps_b, n_output]"""

        x = self.model(x)
        return x