import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers as tfl

# ==================================================================================================


class ConvModule(tfl.Layer):
    def __init__(self, filters: int, kernel_size: int):
        super().__init__()

        self.convp1 = tfl.Conv1D(filters=filters, kernel_size=1, padding="same")
        self.convp2 = tfl.Conv1D(filters=filters, kernel_size=1, padding="same")

        self.convd1 = tfl.DepthwiseConv2D(kernel_size=(kernel_size, 1), padding="same")

        self.lnorm = tfl.LayerNormalization()
        self.bnorm = tfl.BatchNormalization()

    # ==============================================================================================

    @tf.function()
    def call(self, x):  # pylint: disable=arguments-differ
        b = tf.identity(x)

        a = self.lnorm(x)
        a = self.convp1(a)
        a = tf.keras.activations.gelu(a)

        a = tf.expand_dims(a, axis=1)
        a = self.convd1(a)
        a = tf.squeeze(a, axis=1)

        a = self.bnorm(a)
        a = tf.keras.activations.swish(a)
        a = self.convp2(a)
        a = tf.nn.dropout(a, rate=0.1)

        x = tf.add(a, b)
        return x


# ==================================================================================================


class AttentionModule(tfl.Layer):
    def __init__(self, num_heads: int):
        super().__init__()

        self.lnorm = tfl.LayerNormalization()
        self.mharpe = tfl.MultiHeadAttention(num_heads=num_heads, key_dim=2)

    # ==============================================================================================

    # @tf.function()
    def call(self, x):  # pylint: disable=arguments-differ
        b = tf.identity(x)

        a = self.lnorm(x)
        a = self.mharpe(a, a, return_attention_scores=False)
        a = tf.nn.dropout(a, rate=0.1)

        x = tf.add(a, b)
        return x


# ==================================================================================================


class FeedForwardModule(tfl.Layer):
    def __init__(self, filters: int):
        super().__init__()

        self.lnorm = tfl.LayerNormalization()
        self.dense1 = tfl.TimeDistributed(tfl.Dense(filters * 4))
        self.dense2 = tfl.TimeDistributed(tfl.Dense(filters))

    # ==============================================================================================

    # @tf.function()
    def call(self, x):  # pylint: disable=arguments-differ
        b = tf.identity(x)

        a = self.lnorm(x)
        a = self.dense1(x)
        a = tf.keras.activations.swish(a)
        a = tf.nn.dropout(a, rate=0.1)
        a = self.dense2(x)
        a = tf.nn.dropout(a, rate=0.1)

        x = tf.add(a, b)
        return x


# ==================================================================================================


class ConformerBlock(tfl.Layer):
    def __init__(self, filters: int, kernel_size: int, att_heads: int):
        super().__init__()

        self.ff1 = FeedForwardModule(filters)
        self.mha = AttentionModule(att_heads)
        self.conv = ConvModule(filters, kernel_size)
        self.ff2 = FeedForwardModule(filters)
        self.lnorm = tfl.LayerNormalization()

    # ==============================================================================================

    # @tf.function()
    def call(self, x):  # pylint: disable=arguments-differ

        b = tf.identity(x)
        a = self.ff1(x)
        x = tf.add(0.5 * a, b)

        b = tf.identity(x)
        a = self.mha(x)
        x = tf.add(a, b)

        b = tf.identity(x)
        a = self.conv(x)
        x = tf.add(a, b)

        b = tf.identity(x)
        a = self.ff2(x)
        x = tf.add(0.5 * a, b)

        x = self.lnorm(x)
        return x


# ==================================================================================================


class Encoder(Model):  # pylint: disable=abstract-method
    def __init__(self, nlayers: int, dimension: int, att_heads: int):
        super().__init__()

        self.kernel_size = 32
        self.feature_time_reduction_factor = 2
        self.model = self.make_model(nlayers, dimension, att_heads)

    # ==============================================================================================

    def get_time_reduction_factor(self):
        return self.feature_time_reduction_factor

    # ==============================================================================================

    def make_model(self, nlayers: int, dimension: int, att_heads: int):

        model = tf.keras.Sequential(name="Encoder")

        # Convolution subsampling
        conv = tfl.Conv1D(
            filters=dimension, kernel_size=self.kernel_size, strides=2, padding="same"
        )
        model.add(conv)

        # Linear
        dn = tfl.TimeDistributed(tfl.Dense(dimension))
        model.add(dn)

        # Dropout
        dp = tfl.Dropout(rate=0.1)
        model.add(dp)

        # Conformer Blocks
        for _ in range(nlayers):
            cb = ConformerBlock(
                filters=dimension, kernel_size=self.kernel_size, att_heads=att_heads
            )
            model.add(cb)

        return model

    # ==============================================================================================

    def summary(self, line_length=100, **kwargs):  # pylint: disable=arguments-differ
        self.model.summary(line_length=line_length, **kwargs)

    # ==============================================================================================

    def call(self, x):  # pylint: disable=arguments-differ
        x = self.model(x)
        return x


# ==================================================================================================


class MyModel(Model):  # pylint: disable=abstract-method
    def __init__(
        self, c_input: int, c_output: int, nlayers: int, dimension: int, att_heads: int
    ):
        super().__init__()

        self.n_input = c_input
        self.n_output = c_output

        self.encoder = Encoder(nlayers, dimension, att_heads)
        self.feature_time_reduction_factor = self.encoder.get_time_reduction_factor()

        self.model = self.make_model(dimension)

    # ==============================================================================================

    def make_model(self, dimension: int):
        input_tensor = tfl.Input(shape=[None, self.n_input], name="input")

        # Used for easier debugging changes
        x = tf.identity(input_tensor)

        # Encoder for sure different to Conformer paper (https://arxiv.org/pdf/2005.08100.pdf)
        # Paper uses relative positions in the attention layer and the subsampling isn't described
        # Some layers seem to miss too, because Encoder has only half the size of the full paper net
        x = self.encoder(x)

        # As Decoder only a single LSTM layer is used instead of Conformer's RNNT approach
        x = tfl.LSTM(int(dimension * 4), return_sequences=True, stateful=False)(x)
        x = tfl.LayerNormalization()(x)

        # Map to characters
        x = tfl.TimeDistributed(tfl.Dense(self.n_output))(x)

        x = tf.nn.log_softmax(x)
        output_tensor = tf.identity(x, name="output")

        model = Model(input_tensor, output_tensor, name="SimpleConformer")
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
        self.encoder.summary(line_length=line_length, **kwargs)
        print("")
        self.model.summary(line_length=line_length, **kwargs)

    # ==============================================================================================

    # This input signature is required that we can export and load the model in ".pb" format
    # with a variable sequence length, instead of using the one of the first input.
    # The channel value could be fixed, but I didn't find a way to set it to the channels variable.
    @tf.function(input_signature=[tf.TensorSpec([None, None, None], tf.float32)])
    def call(self, x):  # pylint: disable=arguments-differ
        """Call with input shape: [batch_size, steps_a, n_input].
        Outputs a tensor of shape: [batch_size, steps_b, n_output]"""

        x = self.model(x)
        return x
