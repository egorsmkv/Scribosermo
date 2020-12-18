import random

import librosa
import numpy as np
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


def preemphasis(signal, coef=0.97):
    """Emphasizes high-frequency signal components. Doubles pipeline time."""

    def norm(sig):
        sig = sig.numpy().flatten()
        sig = librosa.effects.preemphasis(sig, coef=coef)
        sig = np.expand_dims(sig, axis=1)
        return sig

    signal = tf.py_function(func=norm, inp=[signal], Tout=tf.float32)
    return signal


# ==================================================================================================


def freq_time_mask(spectrogram, param=10):
    # Not working yet
    # spectrogram = tfio.experimental.audio.freq_mask(spectrogram, param=param)
    # spectrogram = tfio.experimental.audio.time_mask(spectrogram, param=param)
    return spectrogram


# ==================================================================================================


def random_speed(
    spectrogram, mean: float, stddev: float, cut_min: float, cut_max: float
):
    """Apply random speed changes, using clipped normal distribution.
    Transforming the spectogram is much faster than transforming the audio signal."""

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
