import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_io as tfio

# ==================================================================================================


def resample(signal, sample_rate, tmp_sample_rate):
    """Resample to given sample rate and back again"""

    signal = tfio.audio.resample(signal, sample_rate, tmp_sample_rate)
    signal = tfio.audio.resample(signal, tmp_sample_rate, sample_rate)
    return signal


# ==================================================================================================


def normalize_volume(signal):
    """Normalize volume to range [-1,1]"""

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


def freq_mask(spectrogram, n=2, max_size=27):
    """See SpecAugment paper - Frequency Masking. Input shape: [1, steps_time, num_bins]"""

    for _ in range(n):
        size = tf.random.uniform(shape=[], maxval=max_size, dtype=tf.int32)
        max_freq = tf.shape(spectrogram)[-1]
        start_mask = tf.random.uniform(shape=[], maxval=max_freq - size, dtype=tf.int32)
        end_mask = start_mask + size

        indices = tf.reshape(tf.range(max_freq), (1, 1, -1))
        condition = tf.math.logical_and(
            tf.math.greater_equal(indices, start_mask), tf.math.less(indices, end_mask)
        )
        spectrogram = tf.where(condition, tf.cast(0, spectrogram.dtype), spectrogram)

    return spectrogram


# ==================================================================================================


def time_mask(spectrogram, n=2, max_size=100):
    """See SpecAugment paper - Time Masking. Input shape: [1, steps_time, num_bins]"""

    for _ in range(n):
        size = tf.random.uniform(shape=[], maxval=max_size, dtype=tf.int32)
        max_freq = tf.shape(spectrogram)[-2]
        start_mask = tf.random.uniform(shape=[], maxval=max_freq - size, dtype=tf.int32)
        end_mask = start_mask + size

        indices = tf.reshape(tf.range(max_freq), (1, -1, 1))
        condition = tf.math.logical_and(
            tf.math.greater_equal(indices, start_mask), tf.math.less(indices, end_mask)
        )
        spectrogram = tf.where(condition, tf.cast(0, spectrogram.dtype), spectrogram)

    return spectrogram


# ==================================================================================================


def spec_cutout(spectrogram, n=5, max_freq_size=27, max_time_size=100):
    """Cut out random patches of the spectrogram"""

    # Temporarily add extra channels dimension and change W and H
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram = tf.transpose(spectrogram, perm=[0, 2, 1, 3])

    width = tf.random.uniform(shape=[], maxval=max_time_size, dtype=tf.int32)
    height = tf.random.uniform(shape=[], maxval=max_freq_size, dtype=tf.int32)

    # Make the random size divisible by 2
    width = tf.cast(width / 2, dtype=tf.int32) * 2
    height = tf.cast(height / 2, dtype=tf.int32) * 2

    for _ in range(n):
        spectrogram = tfa.image.random_cutout(
            spectrogram,
            mask_size=[height, width],
            constant_values=0,
        )

    spectrogram = tf.transpose(spectrogram, perm=[0, 2, 1, 3])
    spectrogram = tf.squeeze(spectrogram, axis=-1)
    return spectrogram


# ==================================================================================================


def spec_dropout(spectrogram, max_rate=0.1):
    """Drops random values of the spectrogram"""

    rate = tf.random.uniform(shape=[], maxval=max_rate, dtype=tf.float32)
    distrib = tf.random.uniform(
        tf.shape(spectrogram), minval=0.0, maxval=1.0, dtype=tf.float32
    )
    mask = 1 - tf.math.floor(distrib + rate)
    spectrogram = spectrogram * mask
    return spectrogram


# ==================================================================================================


def per_feature_norm(features):
    """Normalize features per channel/frequency"""
    f_mean = tf.math.reduce_mean(features, axis=0)
    f_std = tf.math.reduce_std(features, axis=0)

    features = (features - f_mean) / (f_std + 1e-7)
    return features


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
