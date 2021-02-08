import time

import numpy as np
import tflite_runtime.interpreter as tflite

import testing_pb

# ==================================================================================================

checkpoint_file = "/checkpoints/en/qnetp15/exported/model.tflite"
test_wav_path = "/deepspeech-polyglot/extras/exporting/data/test.wav"
alphabet_path = "/deepspeech-polyglot/data/alphabet_en.json"
ds_alphabet_path = "/deepspeech-polyglot/data/alphabet_en.txt"
ds_scorer_path = "/data_prepared/texts/en/kenlm_en.scorer"

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


def timed_transcription(interpreter, wav_path, idx2char, ds_alphabet, ds_scorer):
    """Transcribe an audio file and measure times for intermediate steps"""

    time_start = time.time()

    audio = testing_pb.load_audio(wav_path)
    time_audio = time.time()

    prediction = predict(interpreter, audio)
    time_model = time.time()

    testing_pb.print_prediction_greedy(prediction, idx2char)
    time_greedy = time.time()

    testing_pb.print_prediction_scorer(prediction, ds_alphabet, ds_scorer)
    time_scorer = time.time()

    testing_pb.print_times(
        time_start, time_audio, time_model, time_greedy, time_scorer, wav_path
    )


# ==================================================================================================


def main():
    alphabet, idx2char = testing_pb.load_alphabet_lookup(alphabet_path)
    ds_alphabet, ds_scorer = testing_pb.load_scorer(ds_alphabet_path, ds_scorer_path)

    print("\nLoading model ...")
    interpreter = tflite.Interpreter(model_path=checkpoint_file)
    print("Input details:", interpreter.get_input_details())

    print("Running some initialization steps ...")
    # Run some random predictions to initialize the model
    _ = predict(interpreter, np.random.uniform(-1, 1, [1, 12345]).astype(np.float32))
    _ = predict(interpreter, np.random.uniform(-1, 1, [1, 1234]).astype(np.float32))
    _ = predict(interpreter, np.random.uniform(-1, 1, [1, 123456]).astype(np.float32))

    # Run random decoding step to initialize the scorer
    testing_pb.print_prediction_scorer(
        np.random.uniform(0, 1, [213, 1, len(alphabet) + 1]),
        ds_alphabet,
        ds_scorer,
        print_text=False,
    )

    # Now run the transcription
    print("")
    timed_transcription(interpreter, test_wav_path, idx2char, ds_alphabet, ds_scorer)


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
