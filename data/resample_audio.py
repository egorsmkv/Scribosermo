import argparse
import os

import librosa
import pandas as pd
import tqdm
import soundfile as sf


# ======================================================================================================================

def input_to_output_path(input_path, output_audio_path):
    output_path = os.path.basename(input_path)
    output_path = os.path.join(output_audio_path, output_path)
    return output_path


# ======================================================================================================================

def resample_file(input_path, output_audio_path, output_rate):
    output_path = input_to_output_path(input_path, output_audio_path)

    wav_or, sr = sf.read(input_path)
    wav_rs = librosa.resample(wav_or, sr, output_rate)
    sf.write(output_path, wav_rs, output_rate, sf.default_subtype("wav"))

    file_size = os.path.getsize(output_path)
    return file_size


# ======================================================================================================================

def main():
    parser = argparse.ArgumentParser(description='Resample audiofiles')
    parser.add_argument('--input_csv_path', type=str)
    parser.add_argument('--output_dir_path', type=str)
    parser.add_argument('--output_rate', type=int, default=16000)
    args = parser.parse_args()

    # Keep the german 0 as "null" string
    data = pd.read_csv(args.input_csv_path, keep_default_na=False)

    audio_dir = os.path.join(args.output_dir_path, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    tqdm.tqdm.pandas()

    data["wav_filesize"] = data["wav_filename"].progress_apply(lambda x: resample_file(x, audio_dir, args.output_rate))
    data["wav_filename"] = data["wav_filename"].apply(lambda x: input_to_output_path(x, audio_dir))

    output_path = os.path.basename(args.input_csv_path)
    output_path = os.path.join(args.output_dir_path, output_path)
    data.to_csv(output_path, index=False, encoding='utf-8-sig')


# ======================================================================================================================

if __name__ == '__main__':
    main()
