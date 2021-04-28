# Most of the streaming concept is taken from Nvidia's reverence implementation:
# https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/02_Online_ASR_Microphone_Demo.ipynb

import json
import multiprocessing as mp
import random
import time

import numpy as np
import soundfile as sf
import tflite_runtime.interpreter as tflite

# If you want to improve the transcriptions with an additional language model, without using the
# training container, you can find a prebuilt pip-package in the published assets here:
# https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3
# or for use on a Raspberry Pi you can use the one from extras/misc directory
from ds_ctcdecoder import Alphabet, Scorer, swigwrapper

# ==================================================================================================

# checkpoint_file = "/checkpoints/en/qnetp15/exported/model_streaming.tflite"
checkpoint_file = "/checkpoints/en/qnetp15/exported/model_quantized.tflite"
test_wav_path = "/Scribosermo/extras/exporting/data/test.wav"
alphabet_path = "/Scribosermo/data/en/alphabet.json"
ds_alphabet_path = "/Scribosermo/data/en/alphabet.txt"
ds_scorer_path = "/data_prepared/langmodel/en.scorer"
beam_size = 1024
sample_rate = 16000

# Experiment a little with those values to optimize inference
chunk_size = int(1.0 * sample_rate)
frame_overlap = int(2.0 * sample_rate)
char_offset = 4

with open(alphabet_path, "r", encoding="utf-8") as file:
    alphabet = json.load(file)

ds_alphabet = Alphabet(ds_alphabet_path)
ds_scorer = Scorer(
    alpha=0.931289039105002,
    beta=1.1834137581510284,
    scorer_path=ds_scorer_path,
    alphabet=ds_alphabet,
)
ds_decoder = swigwrapper.DecoderState()
ds_decoder.init(
    alphabet=ds_alphabet,
    beam_size=beam_size,
    cutoff_prob=1.0,
    cutoff_top_n=512,
    ext_scorer=ds_scorer,
    hot_words=dict(),
)

# ==================================================================================================


def predict(interpreter, audio):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Enable dynamic shape inputs
    interpreter.resize_tensor_input(input_details[0]["index"], audio.shape)
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_details[0]["index"], audio)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data


# ==================================================================================================


def load_audio(wav_path):
    audio, _ = sf.read(wav_path, dtype="int16")
    audio = audio / np.iinfo(np.int16).max
    audio = np.expand_dims(audio, axis=0)
    audio = audio.astype(np.float32)
    return audio


# ==================================================================================================


def streamed_transcription(interpreter, wav_path):
    """Transcribe an audio file chunk by chunk"""

    audio = load_audio(wav_path)
    audio = audio[0]

    start = 0
    buffer = np.zeros(shape=2 * frame_overlap + chunk_size, dtype=np.float32)

    # Add some empty padding that the last words are not cut from the transcription
    audio = np.concatenate([audio, np.zeros(shape=frame_overlap, dtype=np.float32)])

    while start < len(audio):
        stop = min(len(audio), start + chunk_size)
        chunk = audio[start:stop]
        start = stop

        # Add new frames to the end of the buffer
        buffer = buffer[chunk_size:]
        buffer = np.concatenate([buffer, chunk])

        # Get prediction for buffer
        ibuffer = np.expand_dims(buffer, axis=0)
        prediction = predict(interpreter, ibuffer)
        prediction = prediction[0]

        # Extract the interesting part in the middle of the prediction
        timesteps_overlap = int(len(prediction) / (len(buffer) / frame_overlap)) - 2
        prediction = prediction[timesteps_overlap:-timesteps_overlap]
        prediction = prediction[: len(prediction) - char_offset]

        # Feed into decoder
        ds_decoder.next(prediction.tolist())

    # Get decoded prediction and convert to text
    results = ds_decoder.decode(num_results=1)
    results = [(res.confidence, ds_alphabet.Decode(res.tokens)) for res in results]
    lm_text = results[0][1]
    print("Prediction scorer: {}".format(lm_text))


# ==================================================================================================


def main():

    print("\nLoading model ...")
    interpreter = tflite.Interpreter(
        model_path=checkpoint_file, num_threads=mp.cpu_count()
    )

    print("Running transcription ...\n")
    streamed_transcription(interpreter, test_wav_path)


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
