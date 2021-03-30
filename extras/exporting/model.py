import math

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers as tfl

import tflite_tools

# ==================================================================================================


class MyModel(Model):  # pylint: disable=abstract-method
    def __init__(self, nn_model, metadata, specmode):
        super().__init__()

        # Spectrogram normalization constants. Taken from:
        # https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/02_Online_ASR_Microphone_Demo.ipynb
        self.feat_norm_fixed_mean = tf.constant(
            [
                -14.95827016,
                -12.71798736,
                -11.76067913,
                -10.83311182,
                -10.6746914,
                -10.15163465,
                -10.05378331,
                -9.53918999,
                -9.41858904,
                -9.23382904,
                -9.46470918,
                -9.56037,
                -9.57434245,
                -9.47498732,
                -9.7635205,
                -10.08113074,
                -10.05454561,
                -9.81112681,
                -9.68673603,
                -9.83652977,
                -9.90046248,
                -9.85404766,
                -9.92560366,
                -9.95440354,
                -10.17162966,
                -9.90102482,
                -9.47471025,
                -9.54416855,
                -10.07109475,
                -9.98249912,
                -9.74359465,
                -9.55632283,
                -9.23399915,
                -9.36487649,
                -9.81791084,
                -9.56799225,
                -9.70630899,
                -9.85148006,
                -9.8594418,
                -10.01378735,
                -9.98505315,
                -9.62016094,
                -10.342285,
                -10.41070709,
                -10.10687659,
                -10.14536695,
                -10.30828702,
                -10.23542833,
                -10.88546868,
                -11.31723646,
                -11.46087382,
                -11.54877829,
                -11.62400934,
                -11.92190509,
                -12.14063815,
                -11.65130117,
                -11.58308531,
                -12.22214663,
                -12.42927197,
                -12.58039805,
                -13.10098969,
                -13.14345864,
                -13.31835645,
                -14.47345634,
            ],
            dtype=tf.float32,
        )
        self.feat_norm_fixed_std = tf.constant(
            [
                3.81402054,
                4.12647781,
                4.05007065,
                3.87790987,
                3.74721178,
                3.68377423,
                3.69344,
                3.54001005,
                3.59530412,
                3.63752368,
                3.62826417,
                3.56488469,
                3.53740577,
                3.68313898,
                3.67138151,
                3.55707266,
                3.54919572,
                3.55721289,
                3.56723346,
                3.46029304,
                3.44119672,
                3.49030548,
                3.39328435,
                3.28244406,
                3.28001423,
                3.26744937,
                3.46692348,
                3.35378948,
                2.96330901,
                2.97663111,
                3.04575148,
                2.89717604,
                2.95659301,
                2.90181116,
                2.7111687,
                2.93041291,
                2.86647897,
                2.73473181,
                2.71495654,
                2.75543763,
                2.79174615,
                2.96076456,
                2.57376336,
                2.68789782,
                2.90930817,
                2.90412004,
                2.76187531,
                2.89905006,
                2.65896173,
                2.81032176,
                2.87769857,
                2.84665271,
                2.80863137,
                2.80707634,
                2.83752184,
                3.01914511,
                2.92046439,
                2.78461139,
                2.90034605,
                2.94599508,
                2.99099718,
                3.0167554,
                3.04649716,
                2.94116777,
            ],
            dtype=tf.float32,
        )

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
        if self.metadata["use_fixed_norm"]:
            features = features - self.feat_norm_fixed_mean
            features = features / self.feat_norm_fixed_std
        else:
            features = self.per_feature_norm(features)

        # Add a name to be able to split the prediction into sub-graphs
        x = tf.identity(features, name="features")

        # Get predictions
        x = tf.expand_dims(x, axis=0)
        x = self.nn_model(x)

        # Prepare for ctc decoding
        x = tf.nn.softmax(x)

        output_tensor = tf.identity(x, name="logits")

        name = "Exported{}".format(self.metadata["network"].title())
        model = Model(input_tensor, output_tensor, name=name)
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

    def summary(self, line_length=100, **kwargs):  # pylint: disable=arguments-differ
        self.model.summary(line_length=line_length, **kwargs)

    # ==============================================================================================

    # This input signature is required that we can export and load the model in ".pb" format
    # with a variable sequence length, instead of using the one of the first input.
    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.float32)])
    def call(self, x):  # pylint: disable=arguments-differ
        """Call with input shape: [1, len_signal], output shape: [len_steps, 1, n_alphabet]"""

        x = self.model(x)
        return x
