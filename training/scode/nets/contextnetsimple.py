import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers as tfl

# ==================================================================================================


class ConvModule(tfl.Layer):
    def __init__(
        self, filters: int, kernel_size: int, stride: int = 1, has_act: bool = True
    ):
        super().__init__()

        self.sconv1d = tfl.SeparableConv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
        )
        self.bnorm = tfl.BatchNormalization()

        if has_act:
            self.act = tf.keras.activations.swish
        else:
            self.act = tf.identity

    # ==============================================================================================

    @tf.function()
    def call(self, x):  # pylint: disable=arguments-differ
        x = self.sconv1d(x)
        x = self.bnorm(x)
        x = self.act(x)
        return x


# ==================================================================================================


class SqueezeExiteModule(tfl.Layer):
    def __init__(self, filters: int, kernel_size: int):
        super().__init__()

        self.conv = ConvModule(filters, kernel_size)
        self.fc1 = tfl.Dense(int(filters / 8))
        self.fc2 = tfl.Dense(filters)

    # ==============================================================================================

    @tf.function()
    def call(self, x):  # pylint: disable=arguments-differ
        a = self.conv(x)
        b = tfl.GlobalAveragePooling1D()(a)
        b = self.fc1(b)
        b = tf.keras.activations.swish(b)
        b = self.fc2(b)
        b = tf.keras.activations.swish(b)
        b = tf.keras.activations.sigmoid(b)
        # Use broadcasting in multiplication instead of tiling beforehand
        b = tf.expand_dims(b, axis=1)
        x = tf.multiply(a, b)
        return x


# ==================================================================================================


class ConvBlock(tfl.Layer):
    def __init__(
        self, nlayers: int, filters: int, kernel_size: int, stride: int, residual: bool
    ):
        super().__init__()
        self.residual = residual

        self.convs_main = []
        for _ in range(nlayers - 1):
            cm = ConvModule(filters, kernel_size, stride=1)
            self.convs_main.append(cm)

        cm = ConvModule(filters, kernel_size, stride=stride)
        self.convs_main.append(cm)

        if self.residual:
            self.conv_skip = ConvModule(
                filters, kernel_size, stride=stride, has_act=False
            )

        self.se = SqueezeExiteModule(filters, kernel_size)

    # ==============================================================================================

    @tf.function()
    def call(self, x):  # pylint: disable=arguments-differ
        b = tf.identity(x)

        for conv in self.convs_main:
            x = conv(x)

        x = self.se(x)

        if self.residual:
            b = self.conv_skip(b)
            x = tf.add(x, b)

        x = tf.keras.activations.swish(x)
        return x


# ==================================================================================================


class Encoder(Model):  # pylint: disable=abstract-method
    def __init__(self, alpha: float):
        super().__init__()

        self.feature_time_reduction_factor = 4
        self.model = self.make_model(alpha)

    # ==============================================================================================

    def get_time_reduction_factor(self):
        return self.feature_time_reduction_factor

    # ==============================================================================================

    @staticmethod
    def make_model(alpha):

        model = tf.keras.Sequential(name="Encoder")

        # C0
        cb = ConvBlock(
            nlayers=1, filters=int(256 * alpha), kernel_size=5, stride=1, residual=False
        )
        model.add(cb)

        # C1-21
        for i in range(1, 22):
            if i in [3, 7]:
                # In the original paper Block14 would have stride=2, too
                stride = 2
            else:
                stride = 1

            if i < 11:
                filters = 256
            else:
                filters = 512

            cb = ConvBlock(
                nlayers=5,
                filters=int(filters * alpha),
                kernel_size=5,
                stride=stride,
                residual=True,
            )
            model.add(cb)

        # C22
        cb = ConvBlock(
            nlayers=1, filters=int(640 * alpha), kernel_size=5, stride=1, residual=False
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
    def __init__(self, c_input: int, c_output: int, alpha: float):
        super().__init__()

        self.n_input = c_input
        self.n_output = c_output

        self.encoder = Encoder(alpha)
        self.feature_time_reduction_factor = self.encoder.get_time_reduction_factor()

        self.model = self.make_model(alpha)

    # ==============================================================================================

    def make_model(self, alpha: float):
        input_tensor = tfl.Input(shape=[None, self.n_input], name="input")

        # Used for easier debugging changes
        x = tf.identity(input_tensor)

        # Encoder as described in ContextNet paper (https://arxiv.org/pdf/2005.03191.pdf)
        # Only the ConvBlock14 has stride=1 instead of stride=2, else the predictions would have
        # less time steps than the labels, which doesn't work for CTC models
        x = self.encoder(x)

        # As Decoder only a single LSTM layer is used instead of ContextNet's RNNT approach
        x = tfl.LSTM(int(1024 * alpha), return_sequences=True, stateful=False)(x)
        x = tfl.LayerNormalization()(x)

        # Map to characters
        x = tfl.TimeDistributed(tfl.Dense(self.n_output))(x)

        x = tf.nn.log_softmax(x)
        output_tensor = tf.identity(x, name="output")

        model = Model(input_tensor, output_tensor, name="ContextNetSimple")
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
