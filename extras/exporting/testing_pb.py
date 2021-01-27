import librosa
import numpy as np
import tensorflow as tf

# ==================================================================================================

checkpoint_dir = "/checkpoints/en/tmp5/exported/pb/"
test_wav = "/deepspeech-polyglot/extras/exporting/data/test.wav"

alphabet = " abcdefghijklmnopqrstuvwxyz'"
idx2char = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant([i for i, u in enumerate(alphabet)]),
        values=tf.constant([u for i, u in enumerate(alphabet)]),
    ),
    default_value=tf.constant(" "),
)

# ==================================================================================================


def print_prediction(prediction):
    logit_lengths = tf.constant(tf.shape(prediction)[0], shape=(1,))
    decoded = tf.nn.ctc_greedy_decoder(prediction, logit_lengths, merge_repeated=True)

    values = tf.cast(decoded[0][0].values, dtype=tf.int32)
    values = idx2char.lookup(values).numpy()
    values = b"".join(values)
    print("Prediction: {}".format(values))


# ==================================================================================================


def main():
    audio, _ = librosa.load(test_wav, sr=16000)
    audio = np.expand_dims(audio, axis=0)

    model = tf.keras.models.load_model(checkpoint_dir)
    model.summary()
    print("Metadata: {}\n".format(model.get_metadata()))

    prediction = model.predict(audio)
    print_prediction(prediction)


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
