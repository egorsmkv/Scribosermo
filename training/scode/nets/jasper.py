import tensorflow as tf
from tensorflow.keras import layers as tfl

# ==================================================================================================


class BaseModule(tfl.Layer):  # pylint: disable=abstract-method
    def __init__(self, filters, kernel_size, dropout, is_last_module=False):
        super().__init__()

        self.is_last_module = is_last_module
        self.dropout = dropout

        self.conv = tfl.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
        )
        self.bnorm = tfl.BatchNormalization()

    # ==============================================================================================

    def call(self, x, training=False):  # pylint: disable=arguments-differ
        x = self.conv(x, training=training)
        x = self.bnorm(x, training=training)

        if not self.is_last_module:
            # Last base module in a block has relu and dropout after the residual connection
            x = tf.keras.activations.relu(x)

            if training:
                x = tf.nn.dropout(x, rate=self.dropout)

        return x


# ==================================================================================================


class PartialBlock(tfl.Layer):  # pylint: disable=abstract-method
    def __init__(self, filters, kernel_size, dropout, repeat):

        super().__init__()
        self.pblocks = []

        # Build block until residual connection
        for _ in range(repeat - 1):
            bm = BaseModule(filters=filters, kernel_size=kernel_size, dropout=dropout)
            self.pblocks.append(bm)

        bm = BaseModule(
            filters=filters,
            kernel_size=kernel_size,
            dropout=dropout,
            is_last_module=True,
        )
        self.pblocks.append(bm)

    # ==============================================================================================

    def call(self, x, training=False):  # pylint: disable=arguments-differ

        for pblock in self.pblocks:
            x = pblock(x, training=training)
        return x


# ==================================================================================================


class MyModel(tf.keras.Model):  # pylint: disable=abstract-method
    def __init__(self, c_input: int, c_output: int, netconfig: dict):
        super().__init__()

        # Check that the netconfig includes all required keys
        reqkeys = {"blocks", "module_repeat", "dense_residuals"}
        assert reqkeys.issubset(set(netconfig.keys())), "Some network keys are missing"

        # Params: output_filters, kernel_size, dropout
        block_params = [
            [256, 11, 0.2],
            [384, 13, 0.2],
            [512, 17, 0.2],
            [640, 21, 0.3],
            [768, 25, 0.4],
        ]
        block_repeat = netconfig["blocks"] / len(block_params)
        assert block_repeat == int(block_repeat)
        block_repeat = int(block_repeat)

        self.n_input = c_input
        self.n_output = c_output
        self.feature_time_reduction_factor = 2

        self.model = self.make_model(
            block_params,
            block_repeat,
            netconfig["module_repeat"],
            netconfig["dense_residuals"],
        )

    # ==============================================================================================

    def make_model(self, block_params, block_repeat, module_repeat, dense_residuals):
        input_tensor = tfl.Input(shape=[None, self.n_input], name="input")

        x = tfl.Conv1D(filters=256, kernel_size=11, strides=2, padding="same")(
            input_tensor
        )
        x = tfl.BatchNormalization()(x)
        x = tfl.ReLU()(x)
        x = tfl.Dropout(0.2)(x)

        residuals = []
        for bparams in block_params:
            for _ in range(block_repeat):
                filters, kernel_size, dropout = bparams

                if not dense_residuals:
                    # Only one residual connection from block to block,
                    # instead of connecting all blocks before
                    residuals = []

                residuals.append(x)
                b = PartialBlock(filters, kernel_size, dropout, module_repeat)(x)

                # Preprocess all the residual inputs,
                # using reversed list for a better readable graph image
                conv_resd = []
                for r in reversed(residuals):
                    c = tfl.Conv1D(filters=filters, kernel_size=1, padding="same")(r)
                    c = tfl.BatchNormalization()(c)
                    conv_resd.append(c)

                conv_resd.append(b)
                x = tfl.Add()(conv_resd)
                x = tfl.ReLU()(x)
                x = tfl.Dropout(dropout)(x)

        x = tfl.Conv1D(filters=896, kernel_size=29, dilation_rate=2, padding="same")(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.ReLU()(x)
        x = tfl.Dropout(0.4)(x)

        x = tfl.Conv1D(filters=1024, kernel_size=1, padding="valid")(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.ReLU()(x)
        x = tfl.Dropout(0.4)(x)

        x = tfl.Conv1D(filters=self.n_output, kernel_size=1, padding="valid")(x)
        output_tensor = tf.identity(x, name="output")

        model = tf.keras.Model(input_tensor, output_tensor, name="Jasper")
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
