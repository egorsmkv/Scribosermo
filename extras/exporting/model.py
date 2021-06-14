import math

import tensorflow as tf
from tensorflow.keras import layers as tfl

import tflite_tools

# ==================================================================================================


class MyModel(tf.keras.Model):  # pylint: disable=abstract-method
    def __init__(self, nn_model, metadata, specmode):
        super().__init__()

        self.metadata: dict = metadata
        self.nn_model = nn_model

        # Create the matrix here, because building it in the make_model function raised errors
        # when loading the exported model
        self.lmw_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.metadata["num_features"],
            num_spectrogram_bins=self.metadata["num_spectrogram_bins"],
            sample_rate=self.metadata["audio_sample_rate"],
            lower_edge_hertz=20,
            upper_edge_hertz=self.metadata["audio_sample_rate"] / 2,
        )

        self.model = self.make_model(specmode)

    # ==============================================================================================

    @staticmethod
    def normalize_volume(signal):
        """Normalize volume to range [-1,1]"""

        gain = 1.0 / (tf.reduce_max(tf.abs(signal)) + 1e-7)
        signal = signal * gain
        return signal

    # ==============================================================================================

    @staticmethod
    def preemphasis(signal, coef=0.97):
        """Emphasizes high-frequency signal components"""

        psig = signal[1:] - coef * signal[:-1]
        signal = tf.concat([[signal[0]], psig], axis=0)
        return signal

    # ==============================================================================================

    @staticmethod
    def per_feature_norm(features):
        """Normalize features per channel/frequency"""
        f_mean = tf.math.reduce_mean(features, axis=0)
        f_std = tf.math.reduce_std(features, axis=0)

        features = (features - f_mean) / (f_std + 1e-7)
        return features

    # ==============================================================================================

    def make_model(self, specmode):
        input_tensor = tfl.Input(shape=[None], name="input_samples")

        # Used for easier debugging changes
        x = tf.identity(input_tensor)

        # Drop batch axis and expand to shape: [len_signal, 1]
        audio = tf.reshape(x[0], [-1, 1])

        # Signal augmentations
        if self.metadata["use_volume_norm"]:
            audio = self.normalize_volume(audio)
        audio = self.preemphasis(audio, coef=0.97)

        # Spectrogram
        if specmode == "pb":
            spectrogram = tf.raw_ops.AudioSpectrogram(
                input=audio,
                window_size=self.metadata["audio_window_samples"],
                stride=self.metadata["audio_step_samples"],
                magnitude_squared=True,
            )
        elif specmode == "tflite":
            # We need a workaround here, because the default spectrogram is not supported in tflite
            # But this doesn't work with normal tf runtime, so both implementations are required
            pcm = tf.squeeze(audio, axis=-1, name="pcm")
            n_fft = 2 ** math.ceil(math.log2(self.metadata["audio_window_samples"]))
            spectrogram = tflite_tools.stft_magnitude_tflite(
                signals=pcm,
                frame_length=int(self.metadata["audio_window_samples"]),
                frame_step=int(self.metadata["audio_step_samples"]),
                fft_length=n_fft,
            )
            spectrogram = tf.expand_dims(spectrogram, axis=0, name="expand_spec")
        else:
            raise ValueError()

        # LogFilterbanks
        mel_spectrograms = tf.tensordot(spectrogram, self.lmw_matrix, 1)
        mel_spectrograms.set_shape(
            spectrogram.shape[:-1].concatenate(self.lmw_matrix.shape[-1:])
        )
        features = tf.math.log(mel_spectrograms + 1e-6)

        # Optionally calculate MFCC features
        if self.metadata["num_mfcc_features"] > 0:
            features = tflite_tools.mfcc_tflite(features)
            features = features[..., : self.metadata["num_mfcc_features"]]

        # Remove batch dimension
        features = features[0]

        # Feature augmentation
        features = self.per_feature_norm(features)

        # Add a name to be able to split the prediction into sub-graphs
        x = tf.identity(features, name="features")

        # Get predictions
        x = tf.expand_dims(x, axis=0)
        x = self.nn_model(x, training=False)

        # Prepare for ctc decoding
        x = tf.nn.softmax(x)

        output_tensor = tf.identity(x, name="logits")

        name = "Exported{}".format(self.metadata["network"].title())
        model = tf.keras.Model(input_tensor, output_tensor, name=name)
        return model

    # ==============================================================================================

    # Input signature is required to export this method into ".pb" format and use it in inference
    @tf.function(input_signature=[])
    def get_time_reduction_factor(self):
        """Some models reduce the time dimension of the features, for example with striding."""
        return self.nn_model.get_time_reduction_factor()

    # ==============================================================================================

    # Input signature is required to export this method into ".pb" format and use it in inference
    @tf.function(input_signature=[])
    def get_metadata(self):
        """Return metadata for model"""
        return self.metadata

    # ==============================================================================================

    def summary(self):  # pylint: disable=arguments-differ
        print("")
        self.model.summary(line_length=100)

    # ==============================================================================================

    # This input signature is required that we can export and load the model in ".pb" format
    # with a variable sequence length, instead of using the one of the first input.
    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
    def call(self, x):  # pylint: disable=arguments-differ
        """Call with input shape: [1, len_signal], output shape: [1, len_steps, n_alphabet]"""

        x = self.model(x)
        return x
