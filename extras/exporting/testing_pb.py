import json
import time

import numpy as np
import soundfile as sf
import tensorflow as tf

# If you want to improve the transcriptions with an additional language model, without using the
# training container, you can find a prebuilt pip-package in the published assets here:
# https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3
# or for use on a Raspberry Pi you can use the one from extras/misc directory
from ds_ctcdecoder import Alphabet, Scorer, ctc_beam_search_decoder

# ==================================================================================================

checkpoint_dir = "/checkpoints/en/qnetp15/exported/pb/"
test_wav_path = "/deepspeech-polyglot/extras/exporting/data/test.wav"
alphabet_path = "/deepspeech-polyglot/data/alphabet_en.json"
ds_alphabet_path = "/deepspeech-polyglot/data/alphabet_en.txt"
ds_scorer_path = "/data_prepared/texts/en/kenlm_en.scorer"

with open(alphabet_path, "r", encoding="utf-8") as file:
    alphabet = json.load(file)
idx2char = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant([i for i, u in enumerate(alphabet)]),
        values=tf.constant([u for i, u in enumerate(alphabet)]),
    ),
    default_value=tf.constant(" "),
)

ds_alphabet = Alphabet(ds_alphabet_path)
ds_scorer = Scorer(
    alpha=0.931289039105002,
    beta=1.1834137581510284,
    scorer_path=ds_scorer_path,
    alphabet=ds_alphabet,
)

# ==================================================================================================


def print_prediction_greedy(prediction):

    tpred = tf.transpose(prediction, perm=[1, 0, 2])
    logit_lengths = tf.constant(tf.shape(tpred)[0], shape=(1,))
    decoded = tf.nn.ctc_greedy_decoder(tpred, logit_lengths, merge_repeated=True)

    values = tf.cast(decoded[0][0].values, dtype=tf.int32)
    values = idx2char.lookup(values).numpy()
    gd_text = b"".join(values).decode("utf-8")
    print("Prediction greedy: {}".format(gd_text))


# ==================================================================================================


def print_prediction_scorer(prediction, print_text=True):

    ldecoded = ctc_beam_search_decoder(
        prediction.tolist(),
        ds_alphabet,
        beam_size=256,
        cutoff_prob=1.0,
        cutoff_top_n=512,
        scorer=ds_scorer,
        hot_words=dict(),
        num_results=1,
    )
    lm_text = ldecoded[0][1]

    if print_text:
        print("Prediction scorer: {}".format(lm_text))


# ==================================================================================================


def load_audio(wav_path):
    audio, _ = sf.read(wav_path, dtype="int16")
    audio = audio / np.iinfo(np.int16).max
    audio = np.expand_dims(audio, axis=0)
    audio = audio.astype(np.float32)
    return audio


# ==================================================================================================


def timed_transcription(model, wav_path):
    """Transcribe an audio file and measure times for intermediate steps"""

    time_start = time.time()

    audio = load_audio(wav_path)
    time_audio = time.time()

    prediction = model.predict(audio)
    time_model = time.time()

    print_prediction_greedy(prediction)
    time_greedy = time.time()

    print_prediction_scorer(prediction[0])
    time_scorer = time.time()

    dur_audio = time_audio - time_start
    dur_model = time_model - time_audio
    dur_greedy = time_greedy - time_model
    dur_scorer = time_scorer - time_greedy

    len_audio = float(sf.info(wav_path).duration)
    msg = "\nLength of audio was {:.3f}s, loading it took {:.3f}s."
    print(msg.format(len_audio, dur_audio))

    msg = "Calculating the predictions did take {:.3f}s, "
    msg += "greedy decoding {:.3f}s and decoding with a scorer {:.3f}s."
    print(msg.format(dur_model, dur_greedy, dur_scorer))

    rtf_greedy = (dur_model + dur_greedy) / len_audio
    rtf_scorer = (dur_model + dur_scorer) / len_audio
    msg = "The Real-Time-Factor is {:.3f} for greedy and {:.3f} for scorer decoding"
    print(msg.format(rtf_greedy, rtf_scorer))


# ==================================================================================================


def main():

    # Load model and print some infos about it
    print("\nLoading model ...")
    model = tf.keras.models.load_model(checkpoint_dir)
    model.summary()
    print("Metadata: {}\n".format(model.get_metadata()))

    print("Running some initialization steps ...")
    # Run some random predictions to initialize the model
    _ = model.predict(np.random.uniform(-1, 1, [1, 12345]))
    _ = model.predict(np.random.uniform(-1, 1, [1, 1234]))
    _ = model.predict(np.random.uniform(-1, 1, [1, 123456]))

    # Run random decoding step to initialize the scorer
    print_prediction_scorer(
        np.random.uniform(0, 1, [213, len(alphabet) + 1]),
        print_text=False,
    )
    print_prediction_scorer(
        np.random.uniform(0, 1, [321, len(alphabet) + 1]),
        print_text=False,
    )

    # Now run the transcription
    print("")
    timed_transcription(model, test_wav_path)


# ==================================================================================================

if __name__ == "__main__":
    main()
