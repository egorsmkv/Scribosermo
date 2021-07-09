import math

import tensorflow as tf
from tensorflow.keras import layers as tfl

# ==================================================================================================


class BaseModule(tfl.Layer):  # pylint: disable=abstract-method
    def __init__(self, filters, kernel_size, has_relu=True):
        super().__init__()

        pad = int(math.floor(kernel_size / 2))
        self.pad1d = tf.keras.layers.ZeroPadding1D(padding=(pad, pad))

        self.sconv1d = tfl.SeparableConv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="valid",
            data_format="channels_last",
            depthwise_regularizer=None,
            pointwise_regularizer=None,
            use_bias=False,
        )

        self.bnorm = tfl.BatchNormalization(momentum=0.9)
        self.has_relu = has_relu

    # ==============================================================================================

    def call(self, x):  # pylint: disable=arguments-differ

        x = self.pad1d(x)
        x = self.sconv1d(x)
        x = self.bnorm(x)

        if self.has_relu:
            # Last base module in a block has the relu after the residual connection
            x = tf.nn.relu(x)

        return x


# ==================================================================================================


class BaseBlock(tf.keras.Model):  # pylint: disable=abstract-method
    def __init__(self, filters, kernel_size, repeat):
        super().__init__()

        self.partial_block = tf.keras.Sequential()
        for _ in range(repeat - 1):
            layer = BaseModule(filters=filters, kernel_size=kernel_size)
            self.partial_block.add(layer)
        layer = BaseModule(filters=filters, kernel_size=kernel_size, has_relu=False)
        self.partial_block.add(layer)

        self.convpt = tfl.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="valid",
            data_format="channels_last",
            kernel_regularizer=None,
            use_bias=False,
        )
        self.bnorm = tfl.BatchNormalization(momentum=0.9)

    # ==============================================================================================

    def call(self, x):  # pylint: disable=arguments-differ
        a = self.partial_block(x)
        b = self.convpt(x)
        b = self.bnorm(b)
        x = tfl.Add()([a, b])
        x = tf.nn.relu(x)
        return x


# ==================================================================================================


class MyModel(tf.keras.Model):  # pylint: disable=abstract-method
    """See Quartznet example config at:
    https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/"""

    def __init__(self, c_input: int, c_output: int, netconfig: dict):
        super().__init__()

        # Check that the netconfig includes all required keys
        reqkeys = {"blocks", "module_repeat"}
        assert reqkeys.issubset(set(netconfig.keys())), "Some network keys are missing"

        block_params = [
            [256, 33],
            [256, 39],
            [512, 51],
            [512, 63],
            [512, 75],
        ]
        block_repeat = netconfig["blocks"] / len(block_params)
        assert block_repeat == int(block_repeat)
        block_repeat = int(block_repeat)

        self.n_input = c_input
        self.n_output = c_output
        self.feature_time_reduction_factor = 2

        if "extra_lstm" in netconfig:
            extra_lstm = netconfig["extra_lstm"]
        else:
            extra_lstm = False

        self.model = self.make_model(
            block_params, block_repeat, netconfig["module_repeat"], extra_lstm
        )

    # ==============================================================================================

    def make_model(self, block_params, block_repeat, module_repeat, extra_lstm):
        input_tensor = tfl.Input(shape=[None, self.n_input], name="input")

        # Used for easier debugging changes
        x = tf.identity(input_tensor)

        # Use manual zero padding instead of "same" padding in the convolution,
        # because else there is an data/weight offset by 1, resulting in wrong outputs
        x = tf.keras.layers.ZeroPadding1D(padding=(16, 16))(x)

        x = tfl.SeparableConv1D(
            filters=256,
            kernel_size=33,
            strides=2,
            padding="valid",
            data_format="channels_last",
            depthwise_regularizer=None,
            pointwise_regularizer=None,
            use_bias=False,
        )(x)

        x = tfl.BatchNormalization(momentum=0.9)(x)
        x = tf.nn.relu(x)
        x = tfl.Dropout(0.1)(x)

        for bparams in block_params:
            for _ in range(block_repeat):
                filters, kernel_size = bparams
                x = BaseBlock(filters, kernel_size, module_repeat)(x)

        x = tf.keras.layers.ZeroPadding1D(padding=(86, 86))(x)
        x = tfl.SeparableConv1D(
            filters=512,
            kernel_size=87,
            dilation_rate=2,
            padding="valid",
            data_format="channels_last",
            depthwise_regularizer=None,
            pointwise_regularizer=None,
            use_bias=False,
        )(x)
        x = tfl.BatchNormalization(momentum=0.9)(x)
        x = tf.nn.relu(x)

        x = tfl.Conv1D(
            filters=1024,
            kernel_size=1,
            padding="valid",
            data_format="channels_last",
            kernel_regularizer=None,
            use_bias=False,
        )(x)
        x = tfl.BatchNormalization(momentum=0.9)(x)
        x = tf.nn.relu(x)

        if extra_lstm:
            # Not described in the paper, but added for an additional experiment
            # To use the pretrained model, update the training code so that the last 5 instead
            # of the last 2 layer weights are newly initialized. Do a frozen training first.
            x = tfl.LSTM(
                int((block_repeat * 150) / 8) * 8, return_sequences=True, stateful=False
            )(x)

        x = tfl.Conv1D(
            filters=self.n_output,
            kernel_size=1,
            padding="valid",
            data_format="channels_last",
            kernel_regularizer=None,
            use_bias=True,
        )(x)

        x = tf.cast(x, dtype="float32")
        x = tf.nn.log_softmax(x)
        output_tensor = tf.identity(x, name="output")

        model = tf.keras.Model(input_tensor, output_tensor, name="Quartznet")
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
