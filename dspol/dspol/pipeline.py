import pandas as pd
import tensorflow as tf

from . import augmentations, utils

# ==================================================================================================

AUTOTUNE = tf.data.experimental.AUTOTUNE
char2idx: tf.lookup.StaticHashTable
audio_sample_rate: int
audio_window_samples: int
audio_step_samples: int
num_features: int

# ==================================================================================================


def initialize(config):
    global char2idx, audio_sample_rate, audio_window_samples, audio_step_samples, num_features

    alphabet = utils.load_alphabet(config)
    char2idx = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant([u for i, u in enumerate(alphabet)]),
            values=tf.constant([i for i, u in enumerate(alphabet)]),
        ),
        default_value=tf.constant(0),
    )

    audio_sample_rate = int(config["audio_sample_rate"])
    feature_type = config["audio_features"]["use_type"]

    num_features = config["audio_features"][feature_type]["num_features"]
    window_len = config["audio_features"][feature_type]["window_len"]
    window_step = config["audio_features"][feature_type]["window_step"]
    audio_window_samples = audio_sample_rate * window_len
    audio_step_samples = audio_sample_rate * window_step


# ==================================================================================================


def text_to_ids(sample):
    text = tf.strings.lower(sample["text"])
    text_as_chars = tf.strings.bytes_split(text)
    text_as_ints = char2idx.lookup(text_as_chars)
    sample["label"] = text_as_ints
    sample["label_length"] = tf.strings.length(text)
    return sample


# ==================================================================================================


def load_audio(sample, augment: bool = False):
    audio_binary = tf.io.read_file(sample["filepath"])
    audio, _ = tf.audio.decode_wav(audio_binary)

    if augment:
        # Run signal augmentations
        audio = augmentations.normalize(audio)
        # audio = augmentations.resample(audio, audio_sample_rate, tmp_sample_rate=8000)
        audio = augmentations.dither(audio, factor=0.00001)
        audio = augmentations.preemphasis(audio, coef=0.97)
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

    # pcm = tf.squeeze(sample["raw_audio"], axis=-1)
    # n_fft = 2 ** math.ceil(math.log2(audio_window_samples))
    # # pad_size = int(n_fft // 2)
    # # pcm = tf.pad(pcm, [[pad_size, pad_size]], "REFLECT")
    # # sample["pcm"] = pcm
    # stfts = tf.signal.stft(
    #     pcm,
    #     frame_length=int(audio_window_samples),
    #     frame_step=int(audio_step_samples),
    #     fft_length=n_fft,
    #     pad_end=False,
    # )
    # spectrogram = tf.abs(stfts)
    # spectrogram = tf.math.pow(spectrogram, 2)
    # spectrogram = tf.expand_dims(spectrogram, axis=0)

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
        dct_coefficient_count=num_features,
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
        num_mel_bins=num_features,
        num_spectrogram_bins=tf.shape(sample["spectrogram"])[-1],
        sample_rate=audio_sample_rate,
        lower_edge_hertz=20,
        # lower_edge_hertz=0,
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
        features = augmentations.per_feature_norm(features)
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
            "text",
            "filepath",
        ]
    }

    return cleaned_sample


# ==================================================================================================


def create_pipeline(
    csv_path: str,
    batch_size: int,
    config: dict,
    augment: bool = False,
    cache_path: str = "",
):
    """Create data-iterator from csv file"""

    # Initialize pipeline values, using config from method call, that we can easily reuse the config
    # from exported checkpoints
    initialize(config)

    # Keep the german 0 as "null" string
    df = pd.read_csv(csv_path, encoding="utf-8", sep="\t", keep_default_na=False)
    df = df[["filepath", "duration", "text"]]

    df = df.sort_values("duration")
    df = df[["filepath", "text"]]
    ds = tf.data.Dataset.from_tensor_slices(dict(df))

    la_func = lambda x: load_audio(x, augment)
    ds = ds.map(map_func=la_func, num_parallel_calls=AUTOTUNE)
    a2s_func = lambda x: audio_to_spect(x, augment)
    ds = ds.map(map_func=a2s_func, num_parallel_calls=AUTOTUNE)

    feature_type = config["audio_features"]["use_type"]
    if feature_type == "mfcc":
        a2m_func = lambda x: audio_to_mfcc(x, augment)
        ds = ds.map(map_func=a2m_func, num_parallel_calls=AUTOTUNE)
    elif feature_type == "lfbank":
        a2b_func = lambda x: audio_to_lfbank(x, augment)
        ds = ds.map(map_func=a2b_func, num_parallel_calls=AUTOTUNE)
    else:
        raise ValueError

    ds = ds.map(text_to_ids, num_parallel_calls=AUTOTUNE)
    ds = ds.map(post_process, num_parallel_calls=AUTOTUNE)

    if batch_size == 1:
        # No need for padding here
        # This also makes debugging easier if the key dropping is skipped
        ds = ds.batch(1)
    else:
        ds = ds.padded_batch(
            batch_size=batch_size,
            drop_remainder=True,
            padded_shapes=(
                {
                    "features": [None, num_features],
                    "feature_length": [],
                    "label": [None],
                    "label_length": [],
                    "text": [],
                    "filepath": [],
                }
            ),
        )

    if cache_path != "":
        ds = ds.cache(cache_path)

    # ds = ds.repeat(50)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
