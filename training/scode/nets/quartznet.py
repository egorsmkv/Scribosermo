import math

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers as tfl

# ==================================================================================================


class BaseModule(Model):
    def __init__(self, filters, kernel_size, has_relu=True):
        super(BaseModule, self).__init__()

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

        self.model = tf.keras.Sequential()
        self.model.add(self.pad1d)
        self.model.add(self.sconv1d)
        self.model.add(tfl.BatchNormalization(momentum=0.9))

        if has_relu:
            # Last base module in a block has the relu after the residual connection
            self.model.add(tfl.ReLU())

    # ==============================================================================================

    def call(self, x):
        x = self.model(x)
        return x


# ==================================================================================================


class BaseBlock(Model):
    def __init__(self, filters, kernel_size, repeat):
        super(BaseBlock, self).__init__()

        self.partial_block = tf.keras.Sequential()
        for i in range(repeat - 1):
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

    @tf.function()
    def call(self, x):
        a = self.partial_block(x)
        b = self.convpt(x)
        b = self.bnorm(b)
        x = tfl.Add()([a, b])
        x = tfl.ReLU()(x)
        return x


# ==================================================================================================


class MyModel(Model):
    """See Quartznet example config at:
    https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/speech2text/"""

    def __init__(self, c_input, c_output, blocks, module_repeat):
        super(MyModel, self).__init__()

        block_params = [
            [256, 33],
            [256, 39],
            [512, 51],
            [512, 63],
            [512, 75],
        ]
        block_repeat = blocks / len(block_params)
        assert block_repeat == int(block_repeat)
        block_repeat = int(block_repeat)

        self.n_input = c_input
        self.n_output = c_output
        self.feature_time_reduction_factor = 2

        self.model = self.make_model(block_params, block_repeat, module_repeat)

    # ==============================================================================================

    def make_model(self, block_params, block_repeat, module_repeat):
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
        x = tfl.ReLU()(x)
        x = tfl.Dropout(0.1)(x)

        for i in range(len(block_params)):
            for j in range(block_repeat):
                filters, kernel_size = block_params[i]
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
        x = tfl.ReLU()(x)

        x = tfl.Conv1D(
            filters=1024,
            kernel_size=1,
            padding="valid",
            data_format="channels_last",
            kernel_regularizer=None,
            use_bias=False,
        )(x)
        x = tfl.BatchNormalization(momentum=0.9)(x)
        x = tfl.ReLU()(x)

        x = tfl.Conv1D(
            filters=self.n_output,
            kernel_size=1,
            padding="valid",
            data_format="channels_last",
            kernel_regularizer=None,
            use_bias=True,
        )(x)
        x = tf.nn.log_softmax(x)
        output_tensor = tf.identity(x, name="output")

        model = Model(input_tensor, output_tensor, name="Quartznet")
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

    def summary(self, line_length=100, **kwargs):
        self.model.summary(line_length=line_length, **kwargs)

    # ==============================================================================================

    # This input signature is required that we can export and load the model in ".pb" format
    # with a variable sequence length, instead of using the one of the first input.
    # The channel value could be fixed, but I didn't find a way to set it to the channels variable.
    @tf.function(input_signature=[tf.TensorSpec([None, None, None], tf.float32)])
    def call(self, x):
        """Call with input shape: [batch_size, steps_a, n_input]. Note that this is different to
        nemo's reference implementation which uses a "channels_first" approach.
        Outputs a tensor of shape: [batch_size, steps_b, n_output]"""

        x = self.model(x)
        return x
