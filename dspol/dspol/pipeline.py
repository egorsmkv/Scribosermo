import glob
import math
import multiprocessing
import os
import time

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

from . import augmentations

# ==================================================================================================

alphabet = " abcdefghijklmnopqrstuvwxyz'"
char2idx = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant([u for i, u in enumerate(alphabet)]),
        values=tf.constant([i for i, u in enumerate(alphabet)]),
    ),
    default_value=tf.constant(0),
)

audio_sample_rate = 16000
AUTOTUNE = tf.data.experimental.AUTOTUNE

num_mfcc_features = 26
feature_win_len = 32
feature_win_step = 20
audio_window_samples = audio_sample_rate * (feature_win_len / 1000)
audio_step_samples = audio_sample_rate * (feature_win_step / 1000)

window_stride = 0.001
window_size = 0.002
num_lfbank_features = 64

# ==================================================================================================


def text_to_ids(sample):
    text = tf.strings.lower(sample["transcript"])
    text_as_chars = tf.strings.bytes_split(text)
    text_as_ints = char2idx.lookup(text_as_chars)
    sample["label"] = text_as_ints
    sample["label_length"] = tf.strings.length(text)
    return sample


# ==================================================================================================


def load_audio(sample, augment: bool = False):
    audio_binary = tf.io.read_file(sample["wav_filename"])
    audio, _ = tf.audio.decode_wav(audio_binary)

    if augment:
        # Run signal augmentations
        # audio = augmentations.resample(audio, tmp_sample_rate=8000)
        # audio = augmentations.preemphasis(audio, coef=0.97)
        pass

    sample["raw_audio"] = audio
    return sample


# ==================================================================================================


def audio_to_spect(sample, augment: bool = False):

    spectrogram = tf.raw_ops.AudioSpectrogram(
        input=sample["raw_audio"],
        window_size=audio_window_samples,
        stride=audio_step_samples,
        magnitude_squared=True,
    )

    if augment:
        # Run spectrogram augmentations
        # spectrogram = augmentations.freq_time_mask(spectrogram)
        # spectrogram = augmentations.random_speed(spectrogram, 1, 0.25, 0.5, 2)
        pass

    sample["spectrogram"] = spectrogram
    return sample


# ==================================================================================================


def audio_to_mfcc(sample, augment: bool = False):

    features = tf.raw_ops.Mfcc(
        spectrogram=sample["spectrogram"],
        sample_rate=audio_sample_rate,
        upper_frequency_limit=audio_sample_rate / 2,
        lower_frequency_limit=20,
        dct_coefficient_count=num_mfcc_features,
    )

    # Drop batch axis
    features = features[0]

    if augment:
        # Run feature augmentations
        pass

    sample["features"] = features
    return sample


# ==================================================================================================


def audio_to_lfbank(sample, augment: bool = False):
    """See: https://www.tensorflow.org/api_docs/python/tf/signal/mfccs_from_log_mel_spectrograms"""

    lmw_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_lfbank_features,
        num_spectrogram_bins=tf.shape(sample["spectrogram"])[-1],
        sample_rate=audio_sample_rate,
        lower_edge_hertz=20,
        upper_edge_hertz=audio_sample_rate / 2,
    )

    mel_spectrograms = tf.tensordot(sample["spectrogram"], lmw_matrix, 1)
    mel_spectrograms.set_shape(
        sample["spectrogram"].shape[:-1].concatenate(lmw_matrix.shape[-1:])
    )

    features = tf.math.log(mel_spectrograms + 1e-6)

    # Drop batch axis
    features = features[0]

    if augment:
        # Run feature augmentations
        pass

    sample["features"] = features
    return sample


# ==================================================================================================


def post_process(sample):

    # A little complicated way to get the real feature tensor shape ...
    mask = tf.ones_like(sample["features"], dtype=tf.int32)
    sample["feature_length"] = tf.reduce_sum(mask, axis=0)[0]

    # Drop unused keys
    cleaned_sample = {
        k: sample[k]
        for k in sample
        if k
        in [
            "features",
            "feature_length",
            "label",
            "label_length",
            "transcript",
            "wav_filename",
        ]
    }

    return cleaned_sample


# ==================================================================================================


def create_pipeline(
    csv_path: str, batch_size: int, feature_type: str, is_training: bool = False
):
    """Create data-iterator from csv file"""

    # Keep the german 0 as "null" string
    df = pd.read_csv(csv_path, sep=",", keep_default_na=False)
    df = df.sort_values("wav_filesize")

    df = df[["wav_filename", "transcript"]]
    ds = tf.data.Dataset.from_tensor_slices(dict(df))

    la_func = lambda x: load_audio(x, is_training)
    ds = ds.map(map_func=la_func, num_parallel_calls=AUTOTUNE)
    a2s_func = lambda x: audio_to_spect(x, is_training)
    ds = ds.map(map_func=a2s_func, num_parallel_calls=AUTOTUNE)

    if feature_type == "mfcc":
        num_channels = num_mfcc_features
        a2m_func = lambda x: audio_to_mfcc(x, is_training)
        ds = ds.map(map_func=a2m_func, num_parallel_calls=AUTOTUNE)
    elif feature_type == "lfbank":
        num_channels = num_lfbank_features
        a2b_func = lambda x: audio_to_lfbank(x, is_training)
        ds = ds.map(map_func=a2b_func, num_parallel_calls=AUTOTUNE)
    else:
        raise ValueError

    ds = ds.map(text_to_ids, num_parallel_calls=AUTOTUNE)
    ds = ds.map(post_process, num_parallel_calls=AUTOTUNE)

    # ds = ds.batch(1)
    ds = ds.padded_batch(
        batch_size=batch_size,
        drop_remainder=True,
        padded_shapes=(
            {
                "features": [None, num_channels],
                "feature_length": [],
                "label": [None],
                "label_length": [],
                "transcript": [],
                "wav_filename": [],
            }
        ),
    )

    # ds = ds.cache("/tmp/dspol_cache")
    ds = ds.repeat(50)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds, num_channels


# ==================================================================================================


def delete_cache():
    """Delete cache files using multiple processes"""

    p = multiprocessing.Pool()
    p.map(os.remove, glob.glob("/tmp/dspol_cache*"))
