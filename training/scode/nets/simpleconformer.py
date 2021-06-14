import tensorflow as tf
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

    def call(self, x, training=False):  # pylint: disable=arguments-differ
        b = tf.identity(x)

        a = self.lnorm(x, training=training)
        a = self.convp1(a, training=training)
        a = tf.keras.activations.gelu(a)

        a = tf.expand_dims(a, axis=1)
        a = self.convd1(a, training=training)
        a = tf.squeeze(a, axis=1)

        a = self.bnorm(a, training=training)
        a = tf.keras.activations.swish(a)
        a = self.convp2(a, training=training)

        if training:
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

    def call(self, x, training=False):  # pylint: disable=arguments-differ
        b = tf.identity(x)

        a = self.lnorm(x, training=training)
        a = self.mharpe(a, a, return_attention_scores=False, training=training)

        if training:
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

    def call(self, x, training=False):  # pylint: disable=arguments-differ
        b = tf.identity(x)

        a = self.lnorm(x, training=training)
        a = self.dense1(a, training=training)
        a = tf.keras.activations.swish(a)

        if training:
            a = tf.nn.dropout(a, rate=0.1)

        a = self.dense2(a, training=training)
        if training:
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

    def call(self, x, training=False):  # pylint: disable=arguments-differ

        b = tf.identity(x)
        a = self.ff1(x, training=training)
        x = tf.add(0.5 * a, b)

        b = tf.identity(x)
        a = self.mha(x, training=training)
        x = tf.add(a, b)

        b = tf.identity(x)
        a = self.conv(x, training=training)
        x = tf.add(a, b)

        b = tf.identity(x)
        a = self.ff2(x, training=training)
        x = tf.add(0.5 * a, b)

        x = self.lnorm(x, training=training)
        return x


# ==================================================================================================


class Encoder(tf.keras.Model):  # pylint: disable=abstract-method
    def __init__(self, nlayers: int, dimension: int, att_heads: int):
        super().__init__()

        self.kernel_size = 32
        self.feature_time_reduction_factor = 2

        # Subsampling layers (both would have stride=2 normally)
        self.conv1 = tfl.SeparableConv1D(
            filters=dimension, kernel_size=self.kernel_size, strides=2, padding="same"
        )
        self.conv2 = tfl.SeparableConv1D(
            filters=dimension, kernel_size=self.kernel_size, strides=1, padding="same"
        )

        # Linear
        self.lin = tfl.TimeDistributed(tfl.Dense(dimension))

        # Conformer Blocks
        self.conf_blocks = []
        for _ in range(nlayers):
            cb = ConformerBlock(
                filters=dimension, kernel_size=self.kernel_size, att_heads=att_heads
            )
            self.conf_blocks.append(cb)

    # ==============================================================================================

    def get_time_reduction_factor(self):
        return self.feature_time_reduction_factor

    # ==============================================================================================

    def call(self, x, training=False):  # pylint: disable=arguments-differ
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.lin(x, training=training)

        if training:
            x = tf.nn.dropout(x, rate=0.1)

        for conf_block in self.conf_blocks:
            x = conf_block(x, training=training)

        return x


# ==================================================================================================


class MyModel(tf.keras.Model):  # pylint: disable=abstract-method
    def __init__(self, c_input: int, c_output: int, netconfig: dict):
        super().__init__()

        # Check that the netconfig includes all required keys
        reqkeys = {"nlayers", "dimension", "attention_heads"}
        assert reqkeys.issubset(set(netconfig.keys())), "Some network keys are missing"

        self.n_input = c_input
        self.n_output = c_output

        self.encoder = Encoder(
            netconfig["nlayers"], netconfig["dimension"], netconfig["attention_heads"]
        )
        self.feature_time_reduction_factor = self.encoder.get_time_reduction_factor()

        self.model = self.make_model(netconfig["dimension"])

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
        x = tfl.LSTM(
            int((dimension * 4.0) / 8) * 8, return_sequences=True, stateful=False
        )(x)
        x = tfl.LayerNormalization()(x)

        # Map to characters
        x = tfl.TimeDistributed(tfl.Dense(self.n_output))(x)

        x = tf.cast(x, dtype="float32")
        x = tf.nn.log_softmax(x)
        output_tensor = tf.identity(x, name="output")

        model = tf.keras.Model(input_tensor, output_tensor, name="SimpleConformer")
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
        self.encoder.summary(line_length=line_length, **kwargs)
        print("")
        self.model.summary(line_length=line_length, **kwargs)

    # ==============================================================================================

    @tf.function(experimental_relax_shapes=True)
    def call(self, x, training=False):  # pylint: disable=arguments-differ
        """Call with input shape: [batch_size, steps_a, n_input].
        Outputs a tensor of shape: [batch_size, steps_b, n_output]"""

        x = self.model(x, training=training)
        return x
