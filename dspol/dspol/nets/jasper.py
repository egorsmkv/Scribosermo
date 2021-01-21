import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers as tfl

# ==================================================================================================


class BaseModule(Model):
    def __init__(self, filters, kernel_size, dropout, is_last_module=False):
        super(BaseModule, self).__init__()

        self.model = tf.keras.Sequential()
        self.model.add(
            tfl.Conv1D(filters=filters, kernel_size=kernel_size, padding="same")
        )
        self.model.add(tfl.BatchNormalization())

        if not is_last_module:
            # Last base module in a block has relu and dropout after the residual connection
            self.model.add(tfl.ReLU())
            self.model.add(tfl.Dropout(rate=dropout))

    # ==============================================================================================

    def call(self, x):
        x = self.model(x)
        return x


# ==================================================================================================


class PartialBlock(Model):
    def __init__(self, filters, kernel_size, dropout, repeat):
        super(PartialBlock, self).__init__()

        # Build block until residual connection
        self.model = tf.keras.Sequential()
        for i in range(repeat - 1):
            l = BaseModule(filters=filters, kernel_size=kernel_size, dropout=dropout)
            self.model.add(l)
        l = BaseModule(filters, kernel_size, dropout, is_last_module=True)
        self.model.add(l)

    # ==============================================================================================

    def call(self, x):
        x = self.model(x)
        return x


# ==================================================================================================


class MyModel(Model):
    def __init__(self, c_input, c_output, blocks, module_repeat, dense_residuals):
        super(MyModel, self).__init__()

        # Params: output_filters, kernel_size, dropout
        block_params = [
            [256, 11, 0.2],
            [384, 13, 0.2],
            [512, 17, 0.2],
            [640, 21, 0.3],
            [768, 25, 0.4],
        ]
        block_repeat = blocks / len(block_params)
        assert block_repeat == int(block_repeat)
        block_repeat = int(block_repeat)

        self.n_input = c_input
        self.n_output = c_output
        self.feature_time_reduction_factor = 2

        self.model = self.make_model(
            block_params, block_repeat, module_repeat, dense_residuals
        )

    # ==============================================================================================

    def make_model(self, block_params, block_repeat, module_repeat, dense_residuals):
        input_tensor = tfl.Input(shape=[None, self.n_input], name="input")

        x = tfl.Conv1D(filters=256, kernel_size=11, strides=2)(input_tensor)
        x = tfl.BatchNormalization()(x)
        x = tfl.ReLU()(x)
        x = tfl.Dropout(0.2)(x)

        residuals = []
        for i in range(len(block_params)):
            for j in range(block_repeat):
                filters, kernel_size, dropout = block_params[i]

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

        x = tfl.Conv1D(filters=896, kernel_size=29, dilation_rate=2)(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.ReLU()(x)
        x = tfl.Dropout(0.4)(x)

        x = tfl.Conv1D(filters=1024, kernel_size=1)(x)
        x = tfl.BatchNormalization()(x)
        x = tfl.ReLU()(x)
        x = tfl.Dropout(0.4)(x)

        x = tfl.Conv1D(filters=self.n_output, kernel_size=1)(x)
        output_tensor = tf.identity(x, name="output")

        model = Model(input_tensor, output_tensor, name="Jasper")
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

    def call(self, x, training=False):
        """Call with input shape: [batch_size, steps_a, n_input],
        outputs tensor of shape: [batch_size, steps_b, n_output]"""

        x = self.model(x)
        return x
