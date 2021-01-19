import tensorflow as tf
import tensorflow_io as tfio

# ==================================================================================================


def resample(signal, sample_rate, tmp_sample_rate):
    """Resample to given sample rate and back again"""

    signal = tfio.audio.resample(signal, sample_rate, tmp_sample_rate)
    signal = tfio.audio.resample(signal, tmp_sample_rate, sample_rate)
    return signal


# ==================================================================================================


def normalize(signal):
    """Normalize signal to range [-1,1]"""

    gain = 1.0 / (tf.reduce_max(tf.abs(signal)) + 1e-7)
    signal = signal * gain
    return signal


# ==================================================================================================


def dither(signal, factor):
    """Amount of additional white-noise dithering to prevent quantization artefacts"""

    signal += factor * tf.random.normal(shape=tf.shape(signal))
    return signal


# ==================================================================================================


def preemphasis(signal, coef=0.97):
    """Emphasizes high-frequency signal components"""

    psig = signal[1:] - coef * signal[:-1]
    signal = tf.concat([[signal[0]], psig], axis=0)
    return signal


# ==================================================================================================


def freq_time_mask(spectrogram, param=10):
    # Not working yet
    # spectrogram = tfio.experimental.audio.freq_mask(spectrogram, param=param)
    # spectrogram = tfio.experimental.audio.time_mask(spectrogram, param=param)
    # tfa.image.random_cutout()
    return spectrogram


# ==================================================================================================


def per_feature_norm(features):
    """Normalize features per channel/frequency"""
    f_mean = tf.math.reduce_mean(features, axis=0)
    f_std = tf.math.reduce_std(features, axis=0)

    spectrogram = (features - f_mean) / (f_std + 1e-7)
    return spectrogram


# ==================================================================================================


def random_speed(
    spectrogram, mean: float, stddev: float, cut_min: float, cut_max: float
):
    """Apply random speed changes, using clipped normal distribution.
    Transforming the spectrogram is much faster than transforming the audio signal."""

    # Get a random speed and clip it, that we don't reach edge cases where the speed is negative
    change = tf.random.normal(shape=[], mean=mean, stddev=stddev)
    change = tf.minimum(tf.maximum(change, cut_min), cut_max)

    old_shape = tf.shape(spectrogram)
    old_time_size = tf.cast(old_shape[1], tf.float32)
    new_time_size = tf.cast(old_time_size / change, tf.int32)

    # Adding a temporary channel dimension for resizing
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = tf.image.resize(spectrogram, [new_time_size, old_shape[2]])
    spectrogram = tf.squeeze(spectrogram, axis=-1)

    return spectrogram
