import math

import librosa
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare
import numpy as np

from dspol import pipeline

# ==================================================================================================

test_csv = "/deepspeech-polyglot/extras/nemo/data/test.csv"
test_wav = "/deepspeech-polyglot/extras/nemo/data/test.wav"

alphabet = " abcdefghijklmnopqrstuvwxyz'"
idx2char = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant([i for i, u in enumerate(alphabet)]),
        values=tf.constant([u for i, u in enumerate(alphabet)]),
    ),
    default_value=tf.constant(" "),
)

# ==================================================================================================


def test_random_input(onnx_path: str):
    onnx_model = onnx.load(onnx_path)
    onnxtf_model = prepare(onnx_model)

    inp = np.random.uniform(low=-1, high=1, size=[1, 64, 123]).astype(np.float32)
    out = onnxtf_model.run(inp)
    print("Result: ", out)
    print("Shape: ", out.logprobs.shape)


# ==================================================================================================


def make_prediction(onnxtf_model, features):
    prediction = onnxtf_model.run(features)
    prediction = prediction.logprobs
    print(prediction)
    print(prediction.shape)

    # Switch batch_size and time_steps
    prediction = tf.transpose(prediction, perm=[1, 0, 2])

    logit_lengths = tf.constant(tf.shape(prediction)[0], shape=(1,))
    decoded = tf.nn.ctc_greedy_decoder(prediction, logit_lengths, merge_repeated=True)
    print(decoded)

    values = tf.cast(decoded[0][0].values, dtype=tf.int32)
    values = idx2char.lookup(values).numpy()
    values = b"".join(values)
    print("Prediction: {}".format(values))


# ==================================================================================================


def test_csv_input(onnx_path: str, csv_path: str):
    onnx_model = onnx.load(onnx_path)
    onnxtf_model = prepare(onnx_model)
    tds, _ = pipeline.create_pipeline(csv_path, 1, "lfbank", is_training=False)

    for samples in tds:
        features = samples["features"]
        features = tf.transpose(features, [0, 2, 1])
        print(features)
        print(features.shape)
        make_prediction(onnxtf_model, features)


# ==================================================================================================


def test_librosa_input(onnx_path: str, audio_path: str):
    onnx_model = onnx.load(onnx_path)
    onnxtf_model = prepare(onnx_model)

    def normalize_signal(signal, gain=None):
        """Normalize float32 signal to [-1, 1] range"""
        if gain is None:
            gain = 1.0 / (np.max(np.abs(signal)) + 1e-5)
        return signal * gain

    signal, sample_freq = librosa.load(audio_path)
    window_size = 20e-3
    window_stride = 10e-3
    num_fft = 2 ** math.ceil(math.log2(window_size * sample_freq))

    signal = normalize_signal(signal, None)
    signal = librosa.effects.preemphasis(signal, coef=0.97)

    sfts = librosa.core.stft(
        signal,
        n_fft=num_fft,
        hop_length=int(window_stride * sample_freq),
        win_length=int(window_size * sample_freq),
        center=True,
        window=np.hanning,
    )
    S = np.square(np.abs(sfts))

    mel_basis = librosa.filters.mel(
        sample_freq, num_fft, n_mels=64, fmin=0, fmax=int(sample_freq / 2)
    )
    features = np.log(np.dot(mel_basis, S) + 1e-20).T

    features = [features]
    features = tf.transpose(features, [0, 2, 1])
    print(features)
    print(features.shape)
    make_prediction(onnxtf_model, features)


# ==================================================================================================


def print_onnx_infos(onnx_path: str):
    onnx_model = onnx.load(onnx_path)
    print("Input:", onnx_model.graph.input)
    print("Output:", onnx_model.graph.output)


# ==================================================================================================

# test_random_input("/checkpoints/model.onnx")
# test_random_input("/nemo/models/QuartzNet5x5LS-En.onnx")
# print_onnx_infos("/nemo/models/QuartzNet5x5LS-En.onnx")
# print_onnx_infos("/checkpoints/model.onnx")
test_csv_input("/nemo/models/QuartzNet5x5LS-En.onnx", test_csv)
# test_librosa_input("/nemo/models/QuartzNet5x5LS-En.onnx", test_wav)
