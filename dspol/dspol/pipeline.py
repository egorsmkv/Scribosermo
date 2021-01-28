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


def apply_augmentations(tensor, datatype: str, config: dict, train_mode: bool = False):
    """Checks which augmentations are selected and applies them"""

    if datatype == "signal":
        augs = config["augmentations"]["signal"]

        if "normalize_volume" in augs:
            aug = augs["normalize_volume"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.normalize_volume(tensor)

        if "resample" in augs:
            aug = augs["resample"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.resample(
                    tensor, audio_sample_rate, aug["tmp_sample_rate"]
                )

        if "dither" in augs:
            aug = augs["dither"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.dither(tensor, aug["factor"])

        if "preemphasis" in augs:
            aug = augs["preemphasis"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.preemphasis(tensor, aug["coefficient"])

    if datatype == "spectrogram":
        augs = config["augmentations"]["spectrogram"]

        if "freq_mask" in augs:
            aug = augs["freq_mask"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.freq_mask(tensor, aug["n"], aug["max_size"])

        if "time_mask" in augs:
            aug = augs["time_mask"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.time_mask(tensor, aug["n"], aug["max_size"])

        if "spec_cutout" in augs:
            aug = augs["spec_cutout"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.spec_cutout(
                    tensor, aug["n"], aug["max_freq_size"], aug["max_time_size"]
                )

        if "spec_dropout" in augs:
            aug = augs["spec_dropout"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.spec_dropout(tensor, aug["max_percentage"])

        if "random_speed" in augs:
            aug = augs["random_speed"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.random_speed(
                    tensor, aug["mean"], aug["stddev"], aug["cut_min"], aug["cut_max"]
                )

    if datatype == "features":
        augs = config["augmentations"]["features"]

        if "normalize_features" in augs:
            aug = augs["normalize_features"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.per_feature_norm(tensor)

        if "random_add" in augs:
            aug = augs["random_add"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.dither(tensor, aug["factor"])

    return tensor


# ==================================================================================================


def load_audio(sample, config: dict, train_mode: bool = False):
    audio_binary = tf.io.read_file(sample["filepath"])
    audio, _ = tf.audio.decode_wav(audio_binary)

    audio = apply_augmentations(audio, "signal", config, train_mode)
    sample["raw_audio"] = audio
    return sample


# ==================================================================================================


def audio_to_spect(sample, config: dict, train_mode: bool = False):

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

    spectrogram = apply_augmentations(spectrogram, "spectrogram", config, train_mode)
    sample["spectrogram"] = spectrogram
    return sample


# ==================================================================================================


def audio_to_mfcc(sample, config: dict, train_mode: bool = False):
    """Calculate MFCC from spectrogram"""

    features = tf.raw_ops.Mfcc(
        spectrogram=sample["spectrogram"],
        sample_rate=audio_sample_rate,
        upper_frequency_limit=audio_sample_rate / 2,
        lower_frequency_limit=20,
        dct_coefficient_count=num_features,
    )

    # Drop batch axis
    features = features[0]

    features = apply_augmentations(features, "features", config, train_mode)
    sample["features"] = features
    return sample


# ==================================================================================================


def audio_to_lfbank(sample, config: dict, train_mode: bool = False):
    """Calculate log mel filterbanks from spectrogram"""

    # See: https://www.tensorflow.org/api_docs/python/tf/signal/mfccs_from_log_mel_spectrograms
    lmw_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_features,
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

    features = apply_augmentations(features, "features", config, train_mode)
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
    train_mode: bool = False,
    cache_path: str = "",
):
    """Create data-iterator from tab separated csv file"""

    # Initialize pipeline values, using config from method call, that we can easily reuse the config
    # from exported checkpoints
    initialize(config)

    # Keep the german 0 as "null" string
    df = pd.read_csv(csv_path, encoding="utf-8", sep="\t", keep_default_na=False)
    df = df[["filepath", "duration", "text"]]

    if config["repeat_train_dataset"] and train_mode is True:
        # Replicate the data multiple times
        df = pd.concat([df] * config["repeat_ds_times"], ignore_index=True)

    if config["sort_datasets"]:
        df = df.sort_values("duration", ascending=config["sort_ds_ascending"])

    df = df[["filepath", "text"]]
    ds = tf.data.Dataset.from_tensor_slices(dict(df))

    la_func = lambda x: load_audio(x, config, train_mode)
    ds = ds.map(map_func=la_func, num_parallel_calls=AUTOTUNE)
    a2s_func = lambda x: audio_to_spect(x, config, train_mode)
    ds = ds.map(map_func=a2s_func, num_parallel_calls=AUTOTUNE)

    feature_type = config["audio_features"]["use_type"]
    if feature_type == "mfcc":
        a2m_func = lambda x: audio_to_mfcc(x, config, train_mode)
        ds = ds.map(map_func=a2m_func, num_parallel_calls=AUTOTUNE)
    elif feature_type == "lfbank":
        a2b_func = lambda x: audio_to_lfbank(x, config, train_mode)
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
