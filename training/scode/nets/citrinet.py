import tensorflow as tf
from tensorflow.keras import layers as tfl

# ==================================================================================================


class ConvModule(tfl.Layer):
    def __init__(
        self, filters: int, kernel_size: int, stride: int = 1, has_act: bool = True
    ):
        super().__init__()
        self.has_act = has_act

        self.sconv1d = tfl.SeparableConv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            padding="same",
        )
        self.bnorm = tfl.BatchNormalization()

    # ==============================================================================================

    def call(self, x, training=False):  # pylint: disable=arguments-differ
        x = self.sconv1d(x, training=training)
        x = self.bnorm(x, training=training)

        if self.has_act:
            x = tf.keras.activations.swish(x)

        return x


# ==================================================================================================


class SqueezeExiteModule(tfl.Layer):
    def __init__(self, filters: int, kernel_size: int):
        super().__init__()

        self.conv = ConvModule(filters, kernel_size)
        self.fc1 = tfl.Dense(int(filters / 8))
        self.fc2 = tfl.Dense(filters)

    # ==============================================================================================

    def call(self, x, training=False):  # pylint: disable=arguments-differ
        a = self.conv(x, training=training)

        b = tfl.GlobalAveragePooling1D()(a)
        b = self.fc1(b, training=training)
        b = tf.keras.activations.swish(b)

        b = self.fc2(b, training=training)
        b = tf.keras.activations.swish(b)
        b = tf.nn.sigmoid(b)

        # Use broadcasting in multiplication instead of tiling beforehand
        b = tf.expand_dims(b, axis=1)
        x = tf.multiply(a, b)
        return x


# ==================================================================================================


class BaseModule(tfl.Layer):  # pylint: disable=abstract-method
    def __init__(self, filters, kernel_size, stride=1, convbn_only=False):
        super().__init__()
        self.convbn_only = convbn_only

        self.sconv1d = tfl.SeparableConv1D(
            filters=filters, kernel_size=kernel_size, padding="same", strides=stride
        )
        self.bnorm = tfl.BatchNormalization(momentum=0.9)

    # ==============================================================================================

    def call(self, x, training=False):  # pylint: disable=arguments-differ

        x = self.sconv1d(x, training=training)
        x = self.bnorm(x, training=training)

        if not self.convbn_only:
            # Last base module in a block has a different structure
            x = tf.nn.relu(x)
            if training:
                x = tf.nn.dropout(x, rate=0.1)

        return x


# ==================================================================================================


class BaseBlock(tfl.Layer):  # pylint: disable=abstract-method
    def __init__(self, filters, kernel_size, stride_first, repeat):
        super().__init__()

        self.pblocks = []
        for i in range(repeat - 1):
            if i == 0 and stride_first != 1:
                bm = BaseModule(
                    filters=filters, kernel_size=kernel_size, stride=stride_first
                )
            else:
                bm = BaseModule(filters=filters, kernel_size=kernel_size, stride=1)
            self.pblocks.append(bm)

        bm = BaseModule(filters=filters, kernel_size=kernel_size, convbn_only=True)
        self.pblocks.append(bm)

        self.convpt = tfl.Conv1D(
            filters=filters, kernel_size=1, padding="same", strides=stride_first
        )
        self.bnorm = tfl.BatchNormalization(momentum=0.9)

        # The same module as in ContextNet paper
        self.se = SqueezeExiteModule(filters, kernel_size)

    # ==============================================================================================

    def call(self, x, training=False):  # pylint: disable=arguments-differ
        a = tf.identity(x)

        for pblock in self.pblocks:
            a = pblock(a, training=training)

        a = self.se(a)

        b = self.convpt(x)
        b = self.bnorm(b)

        x = tf.add(a, b)
        x = tf.nn.relu(x)
        if training:
            x = tf.nn.dropout(x, rate=0.1)

        return x


# ==================================================================================================


class MyModel(tf.keras.Model):  # pylint: disable=abstract-method
    def __init__(self, c_input: int, c_output: int, netconfig: dict):
        super().__init__()

        # Check that the netconfig includes all required keys
        reqkeys = {"channels"}
        assert reqkeys.issubset(set(netconfig.keys())), "Some network keys are missing"

        block_params = [
            [11, 13, 15, 17, 19, 21],
            [13, 15, 17, 19, 21, 23, 25],
            [25, 27, 29, 31, 33, 35, 37, 39],
        ]
        self.megablocks = len(block_params)

        self.n_input = c_input
        self.n_output = c_output
        self.feature_time_reduction_factor = 2

        if "extra_lstm" in netconfig:
            extra_lstm = netconfig["extra_lstm"]
        else:
            extra_lstm = False

        self.model = self.make_model(netconfig["channels"], block_params, extra_lstm)

    # ==============================================================================================

    def make_model(self, filters, block_params, extra_lstm):
        input_tensor = tfl.Input(shape=[None, self.n_input], name="input")

        # Used for easier debugging changes
        x = tf.identity(input_tensor)

        # Prolog
        x = tfl.SeparableConv1D(
            filters=filters,
            kernel_size=5,
            strides=1,
            padding="same",
        )(x)
        x = tfl.BatchNormalization(momentum=0.9)(x)
        x = tf.nn.relu(x)

        # Megablock1-3
        # For some reason this model has more params than the model described in CitriNet paper,
        # which is also different to the released NeMo models
        for i in range(self.megablocks):
            if i in [0]:
                # In the original paper all mega-blocks would have stride=2, but this doesn't work
                # with a character-based CTC decoder
                stride_first = 2
            else:
                stride_first = 1

            # b = tf.identity(x)
            for ksize in block_params[i]:
                x = BaseBlock(filters, ksize, stride_first, repeat=5)(x)
                # Only the first base-block in a mega-block has strides
                stride_first = 1

            # x = tf.add(x, b)

        # Epilog
        x = tfl.SeparableConv1D(
            filters=filters,
            kernel_size=41,
            padding="same",
        )(x)
        x = tfl.BatchNormalization(momentum=0.9)(x)
        x = tf.nn.relu(x)

        if extra_lstm:
            # Not described in the paper, but added for an additional experiment
            x = tfl.LSTM(
                int((filters * 1.75) / 8) * 8, return_sequences=True, stateful=False
            )(x)

        x = tfl.Conv1D(
            filters=self.n_output,
            kernel_size=1,
            padding="valid",
            use_bias=True,
        )(x)

        x = tf.cast(x, dtype="float32")
        x = tf.nn.log_softmax(x)
        output_tensor = tf.identity(x, name="output")

        model = tf.keras.Model(input_tensor, output_tensor, name="CitriNet")
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

        # Run model in eval mode, because the previous trainings have been run like this
        # Enabling training flag would now result in a high loss if using pretrained models
        x = self.model(x, training=False)
        return x
