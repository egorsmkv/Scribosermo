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
    global char2idx

    text = tf.strings.lower(sample["text"])
    text_as_chars = tf.strings.unicode_split(text, "UTF-8")
    text_as_ints = char2idx.lookup(text_as_chars)
    sample["label"] = text_as_ints
    sample["label_length"] = tf.strings.length(text)
    return sample


# ==================================================================================================


def apply_augmentations(tensor, datatype: str, config: dict, train_mode: bool = False):
    """Checks which augmentations are selected and applies them"""

    if datatype == "signal":
        augs = config["augmentations"]["signal"]

        if "dither" in augs:
            aug = augs["dither"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.dither(tensor, aug["factor"])

        if "normalize_volume" in augs:
            aug = augs["normalize_volume"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.normalize_volume(tensor)

        if "preemphasis" in augs:
            aug = augs["preemphasis"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.preemphasis(tensor, aug["coefficient"])

        if "resample" in augs:
            aug = augs["resample"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.resample(
                    tensor, audio_sample_rate, aug["tmp_sample_rate"]
                )

        if "random_volume" in augs:
            aug = augs["random_volume"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.random_volume(
                    tensor, aug["min_dbfs"], aug["min_dbfs"]
                )

        if "reverb" in augs:
            aug = augs["reverb"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.reverb(
                    tensor, audio_sample_rate, aug["delay"], aug["decay"]
                )

    if datatype == "spectrogram":
        augs = config["augmentations"]["spectrogram"]

        if "random_pitch" in augs:
            aug = augs["random_pitch"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.random_pitch(
                    tensor, aug["mean"], aug["stddev"], aug["cut_min"], aug["cut_max"]
                )

        if "random_speed" in augs:
            aug = augs["random_speed"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.random_speed(
                    tensor, aug["mean"], aug["stddev"], aug["cut_min"], aug["cut_max"]
                )

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
                tensor = augmentations.spec_dropout(tensor, aug["max_rate"])

    if datatype == "features":
        augs = config["augmentations"]["features"]

        if "random_multiply" in augs:
            aug = augs["random_multiply"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.random_multiply(
                    tensor, aug["mean"], aug["stddev"]
                )

        if "random_add" in augs:
            aug = augs["random_add"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.dither(tensor, aug["factor"])

        if "normalize_features" in augs:
            aug = augs["normalize_features"]
            if (aug["use_train"] and train_mode) or (
                aug["use_test"] and not train_mode
            ):
                tensor = augmentations.per_feature_norm(tensor)

    return tensor


# ==================================================================================================


def load_audio(sample):
    audio_binary = tf.io.read_file(sample["filepath"])
    audio, _ = tf.audio.decode_wav(audio_binary)

    sample["raw_audio"] = audio
    return sample


# ==================================================================================================


def augment_signal(sample, config: dict, train_mode: bool = False):
    audio = tf.squeeze(sample["raw_audio"], axis=-1)
    audio = apply_augmentations(audio, "signal", config, train_mode)

    audio = tf.expand_dims(audio, axis=-1)
    sample["signal"] = audio
    return sample


# ==================================================================================================


def audio_to_spect(sample, config: dict, train_mode: bool = False):
    global audio_window_samples, audio_step_samples

    spectrogram = tf.raw_ops.AudioSpectrogram(
        input=sample["signal"],
        window_size=audio_window_samples,
        stride=audio_step_samples,
        magnitude_squared=True,
    )

    spectrogram = apply_augmentations(spectrogram, "spectrogram", config, train_mode)
    sample["spectrogram"] = spectrogram
    return sample


# ==================================================================================================


def audio_to_mfcc(sample, config: dict, train_mode: bool = False):
    """Calculate MFCC from spectrogram"""
    global audio_sample_rate, num_features

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
    global audio_sample_rate, num_features

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
    keep_keys = [
        "features",
        "feature_length",
        "label",
        "label_length",
        "text",
        "filepath",
    ]
    cleaned_sample = {k: sample[k] for k in sample if k in keep_keys}

    return cleaned_sample


# ==================================================================================================


def create_pipeline(csv_path: str, batch_size: int, config: dict, mode: str):
    """Create data-iterator from tab separated csv file"""
    global num_features

    # Initialize pipeline values, using config from method call, that we can easily reuse the config
    # from exported checkpoints
    initialize(config)

    # Keep the german 0 as "null" string
    df = pd.read_csv(csv_path, encoding="utf-8", sep="\t", keep_default_na=False)
    df = df[["filepath", "duration", "text"]]

    if config["sort_datasets"]:
        df = df.sort_values("duration", ascending=config["sort_ds_ascending"])

    df = df[["filepath", "text"]]
    ds = tf.data.Dataset.from_tensor_slices(dict(df))

    # Load audio from files
    ds = ds.map(map_func=load_audio, num_parallel_calls=AUTOTUNE)

    # Apply augmentations only in training
    train_mode = bool(mode in ["train"])

    as_func = lambda x: augment_signal(x, config, train_mode)
    ds = ds.map(map_func=as_func, num_parallel_calls=AUTOTUNE)
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

    # LSTM networks seem to have problems with half filled batches
    # Drop them in training and evaluation, but keep them while testing
    drop_remainder = bool(mode in ["train", "eval"])

    if batch_size == 1:
        # No need for padding here
        # This also makes debugging easier if the key dropping is skipped
        ds = ds.batch(1)
    else:
        ds = ds.padded_batch(
            batch_size=batch_size,
            drop_remainder=drop_remainder,
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

    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds
