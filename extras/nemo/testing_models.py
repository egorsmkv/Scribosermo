import itertools
import json
import os

import librosa
import numpy as np
import onnx
import scipy.io.wavfile as wave
import tensorflow as tf
from onnx import TensorProto, helper
from onnx_tf.backend import prepare
from tensorflow.keras import Model

from dspol import nets, pipeline, utils

# ==================================================================================================

test_csv = "/deepspeech-polyglot/extras/nemo/data/test.csv"
test_wav = "/deepspeech-polyglot/extras/nemo/data/test.wav"
qnet_blocks = 5

alphabet = " abcdefghijklmnopqrstuvwxyz'"
idx2char = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant([i for i, u in enumerate(alphabet)]),
        values=tf.constant([u for i, u in enumerate(alphabet)]),
    ),
    default_value=tf.constant(" "),
)

pl_config = {
    "alphabet_path": "/deepspeech-polyglot/data/de/alphabet.json",
    "audio_sample_rate": 16000,
    "audio_features": {
        "use_type": "lfbank",
        "lfbank": {"num_features": 64, "window_len": 0.02, "window_step": 0.01},
    },
    "sort_datasets": False,
    "augmentations": {
        "signal": {
            "normalize_volume": {"use_train": True, "use_test": True},
            "dither": {"use_train": True, "use_test": True, "factor": 1e-05},
            "preemphasis": {"use_train": True, "use_test": True, "coefficient": 0.97},
        },
        "features": {"normalize_features": {"use_train": True, "use_test": True}},
    },
}

# ==================================================================================================


def test_random_input(onnx_path: str):
    onnx_model = onnx.load(onnx_path)
    onnxtf_model = prepare(onnx_model)

    inp = np.random.uniform(low=-1, high=1, size=[1, 64, 123]).astype(np.float32)
    out = onnxtf_model.run(inp)
    print("Result: ", out)
    print("Shape: ", out.logprobs.shape)


# ==================================================================================================


def print_prediction(prediction):
    logit_lengths = tf.constant(tf.shape(prediction)[0], shape=(1,))
    decoded = tf.nn.ctc_greedy_decoder(prediction, logit_lengths, merge_repeated=True)

    values = tf.cast(decoded[0][0].values, dtype=tf.int32)
    values = idx2char.lookup(values).numpy()
    values = b"".join(values)
    print("Prediction: {}".format(values))


# ==================================================================================================


def make_prediction_onnx(onnxtf_model, features):
    prediction = onnxtf_model.run(features)
    prediction = prediction.logprobs
    print(prediction)
    print(prediction.shape)

    # Switch batch_size and time_steps
    prediction = tf.transpose(prediction, perm=[1, 0, 2])

    print_prediction(prediction)


# ==================================================================================================


def test_csv_input(onnx_path: str, csv_path: str):
    onnx_model = onnx.load(onnx_path)
    onnxtf_model = prepare(onnx_model)

    tds = pipeline.create_pipeline(csv_path, 1, pl_config, train_mode=True)
    for samples in tds:
        features = samples["features"]
        print(features)
        print(features.shape)
        features = tf.transpose(features, [0, 2, 1])
        make_prediction_onnx(onnxtf_model, features)


# ==================================================================================================


def print_onnx_infos(onnx_path: str):
    onnx_model = onnx.load(onnx_path)
    print("Input:", onnx_model.graph.input)
    print("Output:", onnx_model.graph.output)


# ==================================================================================================


def transfer_onnx_weights(onnx_path: str):

    # Create tensorflow model
    model = nets.quartznet.MyModel(
        64, len(alphabet) + 1, blocks=qnet_blocks, module_repeat=5
    )
    model.build(input_shape=(None, None, 64))
    model.summary()
    # ws = [w.shape for w in model.get_weights()]
    # print("\n", len(ws), ws)

    # Load onnx model
    onnx_model = onnx.load(onnx_path)

    # Weights are not sorted, so we need to extract the sorting from the layers
    nodes = [t for t in onnx_model.graph.node]
    nodes = [n.input for n in nodes]
    # Drop layers without weights
    nodes = [n for n in nodes if len(n) > 1]
    # Drop ids of input layer
    nodes = [n[1:] for n in nodes]
    # print("\n", len(nodes), nodes)

    # Extract weights and their ids
    weights = [t for t in onnx_model.graph.initializer]
    weights = [(w.name, onnx.numpy_helper.to_array(w)) for w in weights]
    # ws = [(n, w.shape) for n,w in weights]
    # print("\n", len(ws), ws)

    # Sort weights
    nodes = list(itertools.chain.from_iterable(nodes))
    weights = sorted(weights, key=lambda item: nodes.index(item[0]))
    # ws = [(n, w.shape) for n, w in weights]
    # print("\n", len(ws), ws)

    # Transpose weights
    t_weights = []
    for _, w in weights:
        if len(w.shape) == 1:
            t_weights.append(w)
        else:
            if w.shape[1] == 1:
                tw = np.transpose(w, (2, 0, 1))
                t_weights.append(tw)
            else:
                tw = np.transpose(w, (2, 1, 0))
                t_weights.append(tw)
    # ws = [w.shape for w in t_weights]
    # print("\n", len(ws), ws)

    # Finally copy the weights into our tensorflow model
    model.set_weights(t_weights)

    return model


# ==================================================================================================


def build_test_tfmodel(onnx_path: str, csv_path: str, checkpoint_dir: str):

    model = transfer_onnx_weights(onnx_path)
    tds = pipeline.create_pipeline(csv_path, 1, pl_config, train_mode=True)

    for samples in tds:
        features = samples["features"]
        print(features)
        print(features.shape)
        predictions = model.predict(features)
        print(predictions)
        print(predictions.shape)
        predictions = tf.transpose(predictions, perm=[1, 0, 2])
        print_prediction(predictions)

    # Export the model
    if os.path.exists(checkpoint_dir):
        utils.delete_dir(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(model=model)
    save_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=1
    )
    save_manager.save()
    tf.keras.models.save_model(model, checkpoint_dir, include_optimizer=False)

    # Export current config next to the checkpoints
    config = utils.get_config()
    path = os.path.join(checkpoint_dir, "config_export.json")
    with open(path, "w+", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


# ==================================================================================================


def debug_models(onnx_path: str, csv_path: str):
    """Compare outputs layer by layer. Partly taken from:
    https://github.com/onnx/onnx-tensorflow/blob/master/example/test_model_large_stepping.py"""

    log_layer = 6
    # Get onnx layer names from netron website
    layer_map = [
        ("194", "separable_conv1d"),
        ("195", "batch_normalization"),
        ("219", "base_block"),
        ("311", "base_block_4"),
        ("313", "separable_conv1d_26"),
        ("DC3", "conv1d_6"),
        ("logprobs", "tf_op_layer_output"),
    ]

    onnx_model = onnx.load(onnx_path)
    more_outputs = []
    output_to_check = []
    for node in onnx_model.graph.node:
        if node.output[0] == layer_map[log_layer][0]:
            more_outputs.append(
                helper.make_tensor_value_info(
                    node.output[0], TensorProto.FLOAT, (100, 100)
                )
            )
            output_to_check.append(node.output[0])
    onnx_model.graph.output.extend(more_outputs)
    onnxtf_model = prepare(onnx_model)

    tfmodel = transfer_onnx_weights(onnx_path)
    itfmodel = Model(
        inputs=tfmodel.get_layer("Quartznet").input,
        outputs=tfmodel.get_layer("Quartznet")
        .get_layer(layer_map[log_layer][1])
        .output,
    )
    itfmodel.build(input_shape=(None, None, 64))
    itfmodel.summary()

    tds = pipeline.create_pipeline(csv_path, 1, pl_config, train_mode=True)
    for samples in tds:
        features = samples["features"]
        # features = np.zeros(shape=(1, 456, 64), dtype=np.float32)
        # features[0][0][0] = 1
        # # features[0][0][1] = 1
        pfeat = tf.transpose(features, [0, 2, 1])
        print(pfeat)
        print(pfeat.shape)

        intermediate_output = itfmodel.predict(features)
        # Nemo's implementation uses a channel first approach, only the last layer has the channels
        # in the last dimension, check this when debugging intermediate layers
        intermediate_output = tf.identity(intermediate_output)
        # intermediate_output = tf.transpose(intermediate_output, [0, 2, 1])
        print(intermediate_output)
        print(intermediate_output.shape)

        tfeatures = tf.transpose(features, [0, 2, 1])
        my_out = onnxtf_model.run(tfeatures)
        print(my_out[layer_map[log_layer][0]])
        print(my_out[layer_map[log_layer][0]].shape)

        flattf = sorted(intermediate_output.numpy().flatten().tolist())[:10]
        flatox = sorted(my_out[layer_map[log_layer][0]].flatten().tolist())[:10]
        print(flattf)
        print(flatox)


# ==================================================================================================


def lrs_features(audio_path):
    """Taken from:
    https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/data/speech2text/speech_utils.py"""

    def normalize_signal(signal):
        gain = 1.0 / (np.max(np.abs(signal)) + 1e-5)
        return signal * gain

    def preemphasis(signal, coeff=0.97):
        return np.append(signal[0], signal[1:] - coeff * signal[:-1])

    _, signal = wave.read(audio_path)
    signal = normalize_signal(signal.astype(np.float32))
    signal = preemphasis(signal, coeff=0.97)

    # print(signal)
    # print(signal.shape)

    stfts = librosa.core.stft(
        signal,
        n_fft=512,
        hop_length=160,
        win_length=320,
        center=True,
        window=np.hanning,
    )
    # print(stfts.T)
    # print(stfts.T.shape)

    spectrogram = np.abs(stfts) ** 2.0
    # print(spectrogram.T)
    # print(spectrogram.T.shape)

    # Build and apply Mel filter
    mel_basis = librosa.filters.mel(16000, 512, n_mels=64, fmin=0, fmax=int(16000 / 2))
    features = np.log(np.dot(mel_basis, spectrogram) + 1e-20).T

    mean = np.mean(features, axis=0)
    std_dev = np.std(features, axis=0)
    features = (features - mean) / std_dev

    return features


# ==================================================================================================


def debug_input(csv_path):
    tds = pipeline.create_pipeline(csv_path, 1, pl_config, train_mode=True)
    np.set_printoptions(edgeitems=10)

    for samples in tds:
        print(samples)

        afp = samples["filepath"].numpy()[0]
        lfeat = lrs_features(afp)
        print(lfeat)
        print(lfeat.shape)

        break


# ==================================================================================================

# test_random_input("/checkpoints/model.onnx")
# test_random_input("/nemo/models/QuartzNet5x5LS-En.onnx")
# print_onnx_infos("/nemo/models/QuartzNet5x5LS-En.onnx")
# print_onnx_infos("/checkpoints/model.onnx")
# test_csv_input("/nemo/models/QuartzNet5x5LS-En.onnx", test_csv)
# debug_models("/nemo/models/QuartzNet5x5LS-En.onnx", test_csv)
# debug_input(test_csv)

# Also update quartznet block number above if you change the model
build_test_tfmodel(
    "/nemo/models/QuartzNet5x5LS-En.onnx", test_csv, "/checkpoints/en/qnet5/"
)
