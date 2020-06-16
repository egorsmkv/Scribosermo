"""
For setup follow:
https://cloud.google.com/text-to-speech/docs/reference/libraries#client-libraries-resources-python
"""

import os
import random
import time

import tqdm
from google.cloud import texttospeech

# ==================================================================================================

# Export application credentials
gac_path = "/DeepSpeech/deepspeech-german/data/google_application_credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gac_path

speakers = [
    "de-DE-Wavenet-A",
    "de-DE-Wavenet-B",
    "de-DE-Wavenet-C",
    "de-DE-Wavenet-D",
    "de-DE-Wavenet-E",
]

client: texttospeech.TextToSpeechClient

text_data_path = "/DeepSpeech/data_original/German_sentences_8mil_filtered_maryfied.txt"
data_directory = "/DeepSpeech/data_original/google_wavenet/"
audio_path = data_directory + "audio/"
csv_path = data_directory + "all.csv"

max_allowed_chars = 1000000


# ==================================================================================================


def download(text, path):
    # Set the text input to be synthesized
    synthesis_input = texttospeech.types.SynthesisInput(text=text)
    speaker = random.choice(speakers)

    # Build the voice request
    voice = texttospeech.types.VoiceSelectionParams(language_code="de-DE", name=speaker)

    # Select the type of audio file you want returned
    audio_config = texttospeech.types.AudioConfig(
        audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
    )

    # Perform the text-to-speech request
    response = client.synthesize_speech(synthesis_input, voice, audio_config)

    sound_path = path + "_" + speaker + ".wav"
    with open(sound_path, "wb") as out:
        # Write the binary response to the output file.
        out.write(response.audio_content)

    file_size = os.path.getsize(sound_path)
    info = sound_path + "," + str(file_size) + "," + text
    return info


# ==================================================================================================


def main():
    global client

    # Instantiates a client
    client = texttospeech.TextToSpeechClient()

    if not os.path.isdir(data_directory):
        os.mkdir(data_directory)
        os.mkdir(audio_path)
        with open(csv_path, "w+", encoding="utf-8") as file:
            file.write("wav_filename,wav_filesize,transcript" + "\n")

    # Read the text data
    with open(text_data_path, encoding="utf-8") as file:
        # Read line by line and remove whitespace characters at the end of each line
        text_data = file.readlines()
        text_data = [x.strip() for x in text_data]

    # Select only some part of the data
    line_start = 60000
    line_end = 60000
    text_data = text_data[line_start:line_end]

    num_of_chars = sum(len(i) for i in text_data)
    print("Number of chars in text data:", num_of_chars)

    if num_of_chars > max_allowed_chars:
        print("List has more chars than allowed")
        exit()

    for i, t in enumerate(tqdm.tqdm(text_data)):
        info = download(t, audio_path + str(i + line_start))

        # Only 300 requests per min
        time.sleep(0.02)

        with open(csv_path, "a", encoding="utf-8") as file:
            file.write(info + "\n")


# ==================================================================================================

if __name__ == "__main__":
    main()
