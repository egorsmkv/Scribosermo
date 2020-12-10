import glob
import multiprocessing
import os
import time

import pandas as pd
import tensorflow as tf
import tqdm

# ==================================================================================================

alphabet = " abcdefghijklmnopqrstuvwxyz'"
char2idx = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant([u for i, u in enumerate(alphabet)]),
        values=tf.constant([i for i, u in enumerate(alphabet)]),
    ),
    default_value=tf.constant(0),
)

n_input = 26
feature_win_len = 32
feature_win_step = 20
audio_sample_rate = 16000
AUTOTUNE = tf.data.experimental.AUTOTUNE

audio_window_samples = audio_sample_rate * (feature_win_len / 1000)
audio_step_samples = audio_sample_rate * (feature_win_step / 1000)


# ==================================================================================================


def get_raw_audio(file_path):
    audio_binary = tf.io.read_file(file_path)
    audio, _ = tf.audio.decode_wav(audio_binary)
    return audio


# ==================================================================================================


def text_to_ids(sample):
    text = tf.strings.lower(sample["transcript"])
    text_as_chars = tf.strings.bytes_split(text)
    text_as_ints = char2idx.lookup(text_as_chars)
    sample["label"] = text_as_ints
    sample["label_length"] = tf.strings.length(text)
    return sample


# ==================================================================================================


def audio_to_mfcc(sample, is_training: bool = False):
    audio = get_raw_audio(sample["wav_filename"])

    if is_training:
        # Run signal augmentations
        pass

    spectrogram = tf.raw_ops.AudioSpectrogram(
        input=audio,
        window_size=audio_window_samples,
        stride=audio_step_samples,
        magnitude_squared=True,
    )

    if is_training:
        # Run spectrogram augmentations
        pass

    features = tf.raw_ops.Mfcc(
        spectrogram=spectrogram,
        sample_rate=audio_sample_rate,
        upper_frequency_limit=audio_sample_rate / 2,
        dct_coefficient_count=n_input,
    )

    if is_training:
        # Run feature augmentations
        pass

    # Drop batch axis
    features = features[0]
    sample["features"] = features

    # A little complicated way to get the real tensor shape ...
    mask = tf.ones_like(features, dtype=tf.int32)
    sample["feature_length"] = tf.reduce_sum(mask, axis=0)[0]

    return sample


# ==================================================================================================


def create_pipeline(csv_path: str, batch_size: int, is_training: bool = False):
    """Create data-iterator from csv file"""

    # Keep the german 0 as "null" string
    df = pd.read_csv(csv_path, sep=",", keep_default_na=False)
    df = df.sort_values("wav_filesize")

    df = df[["wav_filename", "transcript"]]
    ds = tf.data.Dataset.from_tensor_slices(dict(df))

    ds = ds.map(text_to_ids, num_parallel_calls=AUTOTUNE)
    a2m_func = lambda x: audio_to_mfcc(x, is_training)
    ds = ds.map(map_func=a2m_func, num_parallel_calls=AUTOTUNE)

    ds = ds.padded_batch(
        batch_size=batch_size,
        drop_remainder=True,
        padded_shapes=(
            {
                "features": [None, n_input],
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
    return ds


# ==================================================================================================


def delete_cache():
    """Delete cache files using multiple processes"""

    p = multiprocessing.Pool()
    p.map(os.remove, glob.glob("/tmp/dspol_cache*"))


# ==================================================================================================


def test_processing_duration(sample_csv_path, batch_size):
    """ Run pipeline for one epoch to check how long preprocessing takes """

    print("Going through dataset to check preprocessing duration...")
    tds = create_pipeline(sample_csv_path, batch_size)
    for samples in tqdm.tqdm(tds):
        pass


# ==================================================================================================

if __name__ == "__main__":
    sample_csv_path = "/data_prepared/de/voxforge/test_azce.csv"
    delete_cache()

    # test_processing_duration(sample_csv_path, batch_size=24)
    # delete_cache()

    tds = create_pipeline(sample_csv_path, batch_size=2)
    for samples in tds:
        print(samples)
        print(samples["features"].shape[1])
        break
