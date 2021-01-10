import os

import tensorflow as tf
from ds_ctcdecoder import ctc_beam_search_decoder, Scorer, Alphabet

from . import pipeline, training, utils

# ==================================================================================================

# tf.config.optimizer.set_jit(True)

# Allow growing gpu memory
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

config = utils.get_config()
checkpoint_dir = config["checkpoint_dir"]
model: tf.keras.Model

ds_alphabet = Alphabet("/deepspeech-polyglot/data/alphabet_en.txt")
ds_scorer = Scorer(
    alpha=0.931289039105002,
    beta=1.1834137581510284,
    scorer_path="/data_prepared/lm/ds_en.scorer",
    alphabet=ds_alphabet,
)

# ==================================================================================================


def levenshtein(a, b):
    """Calculates the Levenshtein distance between a and b.
    Copied from: hetland.org/coding/python/levenshtein.py"""
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]


# ==================================================================================================


def get_texts(predictions, samples):

    twfs = []
    for i, pred in enumerate(predictions):

        # Calculate text using language model
        lpred = tf.nn.softmax(pred).numpy().tolist()
        ldecoded = ctc_beam_search_decoder(
            lpred,
            ds_alphabet,
            beam_size=1024,
            cutoff_prob=1.0,
            cutoff_top_n=300,
            scorer=ds_scorer,
            hot_words=dict(),
            num_results=1,
        )
        lm_text = ldecoded[0][1]

        # Calculate greedy text
        gpred = tf.expand_dims(pred, axis=1)
        logit_lengths = tf.constant(tf.shape(gpred)[0], shape=(1,))
        gdecoded, _ = tf.nn.ctc_greedy_decoder(
            gpred, logit_lengths, merge_repeated=True
        )
        values = tf.cast(gdecoded[0].values, dtype=tf.int32)
        values = training.idx2char.lookup(values).numpy()
        greedy_text = b"".join(values).decode("utf-8")

        # Get label
        label = samples["label"][i]
        label = training.idx2char.lookup(label).numpy()
        label = b"".join(label).strip().decode("utf-8")

        # Calculate CER and WER of both predictions
        gd_cer = levenshtein(label, greedy_text) / len(label)
        gd_wer = levenshtein(label.split(), greedy_text.split()) / len(label.split())
        lm_cer = levenshtein(label, lm_text) / len(label)
        lm_wer = levenshtein(label.split(), lm_text.split()) / len(label.split())

        # Calculate loss
        samp = {
            "label": [samples["label"][i]],
            "label_length": [samples["label_length"][i]],
        }
        bpred = tf.expand_dims(pred, axis=0)
        loss = training.get_loss(bpred, samp).numpy()

        twf = {
            "filepath": samples["wav_filename"][i].numpy(),
            "label": label,
            "loss": loss,
            "greedy_text": greedy_text,
            "greedy_cer": gd_cer,
            "greedy_wer": gd_wer,
            "lm_text": lm_text,
            "lm_cer": lm_cer,
            "lm_wer": lm_wer,
        }
        twfs.append(twf)

    return twfs


# ==================================================================================================


def run_test(dataset_test):
    print("\nEvaluating ...")
    step = 0
    log_greedy_steps = config["log_prediction_steps"]

    for samples in dataset_test:
        features = samples["features"]
        predictions = model.predict(features)
        step += 1

        if log_greedy_steps != 0 and step % log_greedy_steps == 0:
            training.log_greedy_text(predictions, samples)

        greedy_texts = get_texts(predictions, samples)
        print(greedy_texts)


# ==================================================================================================


def main():
    global model

    # Use exported config to set up the pipeline
    path = os.path.join(checkpoint_dir, "config_export.json")
    exported_config = utils.load_json_file(path)

    dataset_test = pipeline.create_pipeline(
        csv_path=config["data_paths"]["test"],
        batch_size=config["batch_sizes"]["test"],
        config=exported_config,
        augment=True,
        cache_path="",
    )

    # Build model and print network summary
    model = tf.keras.models.load_model(checkpoint_dir)
    feature_type = exported_config["audio_features"]["use_type"]
    c_input = exported_config["audio_features"][feature_type]["num_features"]
    model.build(input_shape=(None, None, c_input))
    model.compile()
    model.summary()

    run_test(dataset_test)
