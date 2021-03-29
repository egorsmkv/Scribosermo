import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_io as tfio

# ==================================================================================================


def dither(signal, factor):
    """Amount of additional white-noise dithering to prevent quantization artefacts"""

    signal += factor * tf.random.normal(shape=tf.shape(signal))
    return signal


# ==================================================================================================


def normalize_volume(signal):
    """Normalize volume to range [-1,1]"""

    gain = 1.0 / (tf.reduce_max(tf.abs(signal)) + 1e-7)
    signal = signal * gain
    return signal


# ==================================================================================================


def preemphasis(signal, coef=0.97):
    """Emphasizes high-frequency signal components"""

    psig = signal[1:] - coef * signal[:-1]
    signal = tf.concat([[signal[0]], psig], axis=0)
    return signal


# ==================================================================================================


def resample(signal, sample_rate, tmp_sample_rate):
    """Resample to given sample rate and back again"""

    signal = tfio.audio.resample(signal, sample_rate, tmp_sample_rate)
    signal = tfio.audio.resample(signal, tmp_sample_rate, sample_rate)
    return signal


# ==================================================================================================


def random_volume(signal, min_dbfs: float, max_dbfs: float):
    """Apply random volume changes.
    Inspired by: DeepSpeech/training/deepspeech_training/util/augmentations.py"""

    def rms_to_dbfs(rms):
        dbfs = tf.math.log(tf.maximum(1e-16, rms)) / tf.math.log(10.0)
        dbfs = 20.0 * dbfs + 3.0103
        return dbfs

    def gain_db_to_ratio(gain_db):
        return tf.math.pow(10.0, gain_db / 20.0)

    def get_max_dbfs(data):
        # Peak dBFS based on the maximum energy sample.
        # Will prevent overdrive if used for normalization.
        amax = tf.abs(tf.reduce_max(data))
        amin = tf.abs(tf.reduce_min(data))
        dmax = tf.maximum(amax, amin)
        dbfs_max = rms_to_dbfs(dmax)
        return dbfs_max

    def normalize(data, dbfs):
        # Change volume and clip signal range
        data = data * gain_db_to_ratio(dbfs - get_max_dbfs(data))
        data = tf.maximum(tf.minimum(data, 1.0), -1.0)
        return data

    target_dbfs = tf.random.uniform(shape=[], minval=min_dbfs, maxval=max_dbfs)
    signal = normalize(signal, target_dbfs)

    return signal


# ==================================================================================================


def reverb(signal, audio_sample_rate, delay: float = 20, decay: float = 10):
    """Adds simplified (no all-pass filters) Schroeder reverberation. Very time consuming.
    Inspired by: DeepSpeech/training/deepspeech_training/util/augmentations.py"""

    def rms_to_dbfs(rms):
        dbfs = tf.math.log(tf.maximum(1e-16, rms)) / tf.math.log(10.0)
        dbfs = 20.0 * dbfs + 3.0103
        return dbfs

    def gain_db_to_ratio(gain_db):
        return tf.math.pow(10.0, gain_db / 20.0)

    def max_dbfs(data):
        # Peak dBFS based on the maximum energy sample.
        # Will prevent overdrive if used for normalization.
        amax = tf.abs(tf.reduce_max(data))
        amin = tf.abs(tf.reduce_min(data))
        dmax = tf.maximum(amax, amin)
        dbfs_max = rms_to_dbfs(dmax)
        return dbfs_max

    def normalize(data, dbfs):
        # Change volume and clip signal range
        data = data * gain_db_to_ratio(dbfs - max_dbfs(data))
        data = tf.maximum(tf.minimum(data, 1.0), -1.0)
        return data

    # Get a random delay and decay
    delay = tf.random.uniform(shape=[], minval=0, maxval=delay)
    decay = tf.random.uniform(shape=[], minval=0, maxval=decay)

    orig_dbfs = max_dbfs(signal)
    decay = gain_db_to_ratio(-decay)
    result = tf.identity(signal)
    signal_len = tf.shape(signal)[0]

    # Primes to minimize comb filter interference
    primes = [17, 19, 23, 29, 31]

    for delay_prime in primes:
        layer = tf.identity(signal)

        # 16 samples minimum to avoid performance trap and risk of division by zero
        n_delay = delay * (delay_prime / primes[0]) * (audio_sample_rate / 1000.0)
        n_delay = tf.cast(tf.floor(tf.maximum(16.0, n_delay)), dtype=tf.int32)

        w_range = tf.cast(tf.floor(signal_len / n_delay), dtype=tf.int32)
        for w_index in range(0, w_range):
            w1 = w_index * n_delay
            w2 = (w_index + 1) * n_delay

            # Last window could be smaller than others
            width = tf.minimum(signal_len - w2, n_delay)

            new_vals = layer[w2 : w2 + width] + decay * layer[w1 : w1 + width]
            layer = tf.concat([layer[:w2], new_vals, layer[w2 + width :]], axis=0)

        result += layer

    signal = normalize(result, dbfs=orig_dbfs)
    return signal


# ==================================================================================================


def random_pitch(
    spectrogram, mean: float, stddev: float, cut_min: float, cut_max: float
):
    """Apply random pitch changes, using clipped normal distribution.
    Inspired by: DeepSpeech/training/deepspeech_training/util/augmentations.py"""

    # Get a random pitch and clip it, that we don't reach edge cases where the speed is negative
    pitch = tf.random.normal(shape=[], mean=mean, stddev=stddev)
    pitch = tf.minimum(tf.maximum(pitch, cut_min), cut_max)

    old_shape = tf.shape(spectrogram)
    old_freq_size = tf.cast(old_shape[2], tf.float32)
    new_freq_size = tf.cast(old_freq_size * pitch, tf.int32)

    # Adding a temporary channel dimension for resizing
    spectrogram = tf.expand_dims(spectrogram, -1)
    spectrogram = tf.image.resize(spectrogram, [old_shape[1], new_freq_size])

    # Crop or pad that we get same number of bins as before
    if pitch > 1:
        spectrogram = tf.image.crop_to_bounding_box(
            spectrogram,
            offset_height=0,
            offset_width=0,
            target_height=old_shape[1],
            target_width=tf.math.minimum(old_shape[2], new_freq_size),
        )
    elif pitch < 1:
        spectrogram = tf.image.pad_to_bounding_box(
            spectrogram,
            offset_height=0,
            offset_width=0,
            target_height=tf.shape(spectrogram)[1],
            target_width=old_shape[2],
        )

    spectrogram = tf.squeeze(spectrogram, axis=-1)
    return spectrogram


# ==================================================================================================


def random_speed(
    spectrogram, mean: float, stddev: float, cut_min: float, cut_max: float
):
    """Apply random speed changes, using clipped normal distribution.
    Transforming the spectrogram is much faster than transforming the audio signal.
    Inspired by: DeepSpeech/training/deepspeech_training/util/augmentations.py"""

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


# ==================================================================================================


def freq_mask(spectrogram, n=2, max_size=27):
    """See SpecAugment paper - Frequency Masking. Input shape: [1, steps_time, num_bins]
    Taken from: https://www.tensorflow.org/io/api_docs/python/tfio/experimental/audio/freq_mask"""

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
    """See SpecAugment paper - Time Masking. Input shape: [1, steps_time, num_bins]
    Taken from: https://www.tensorflow.org/io/api_docs/python/tfio/experimental/audio/time_mask"""

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


def random_multiply(features, mean: float, stddev: float):
    """Add multiplicative random noise to features"""

    features *= tf.random.normal(shape=tf.shape(features), mean=mean, stddev=stddev)
    return features


# ==================================================================================================


def per_feature_norm(features):
    """Normalize features per channel/frequency"""
    f_mean = tf.math.reduce_mean(features, axis=0)
    f_std = tf.math.reduce_std(features, axis=0)

    features = (features - f_mean) / (f_std + 1e-7)
    return features
