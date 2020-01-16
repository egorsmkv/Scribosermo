#! /usr/bin/env python

import argparse

import librosa
import numpy as np
import pandas as pd

# ======================================================================================================================

replacer = {
    "ä": "ae",
    "ö": "oe",
    "ü": "ue"
}

excluded = [
    "/DeepSpeech/data_original/mailabs/by_book/by_book/female/angela_merkel/merkel_alone/wavs/Kanzlerin_11_13_f000060.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000092_0000000873_368740_370300.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000553_0000001098_3357560_3359530.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000106_0000000752_1232380_1234450.wav",
    "/DeepSpeech/data_prepared/common_voice/audio/common_voice_de_17938027.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000169_0000000847_1113250_1116770.wav",
    "/DeepSpeech/data_original/tuda/train/2014-03-17-13-09-27_Microsoft-Kinect-Raw.wav",
    "/DeepSpeech/data_original/tuda/train/2014-03-17-13-03-57_Realtek.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000031_0000001068_1251460_1255320.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000053_0000000081_9487010_9489750.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000031_0000001068_1266510_1270030.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-14-04_Realtek.wav"
]


# ======================================================================================================================

def get_duration(filename):
    """ Get duration of the wav file """
    length = librosa.get_duration(filename=filename)
    return length


# ======================================================================================================================

def print_statistics(data):
    print("")
    print("Duration statistics --- AVG: %.2f   STD: %.2f   MIN: %.2f   MAX: %.2f" % (
        data["duration"].mean(), data["duration"].std(), data["duration"].min(), data["duration"].max()))
    print("Text length statistics --- AVG: %.2f   STD: %.2f   MIN: %.2f   MAX: %.2f" % (
        data["text_length"].mean(), data["text_length"].std(), data["text_length"].min(),
        data["text_length"].max()))
    print("Time per char statistics --- AVG: %.3f   STD: %.3f   MIN: %.3f   MAX: %.3f" % (
        data["avg_time_per_char"].mean(), data["avg_time_per_char"].std(), data["avg_time_per_char"].min(),
        data["avg_time_per_char"].max()))
    print("")


# ======================================================================================================================

def clean(data):
    # Create some temporary columns
    data["duration"] = data.apply(lambda x: get_duration(x.wav_filename), axis=1)
    data["text_length"] = data.apply(lambda x: len(x.transcript), axis=1)
    data["avg_time_per_char"] = data["duration"] / data["text_length"]
    print_statistics(data)

    # Keep only files longer than 1 second
    length_old = len(data)
    data = data[data["duration"] > 1]
    print("Excluded", length_old - len(data), "files with too short duration")

    # Keep only files less than 45 seconds
    length_old = len(data)
    data = data[data["duration"] < 45]
    print("Excluded", length_old - len(data), "files with too long duration")

    # Drop files spoken to fast
    length_old = len(data)
    avg_time = data["avg_time_per_char"].mean()
    data = data[data["avg_time_per_char"] > avg_time / 3]
    print("Excluded", length_old - len(data), "files with too fast char rate")

    # Drop files with a char rate below a second
    length_old = len(data)
    data = data[data["avg_time_per_char"] < 1]
    print("Excluded", length_old - len(data), "files with a much too slow char rate")

    # Keep only files which are not to slowly spoken, except for very short files which may have longer pauses
    length_old = len(data)
    avg_time = data["avg_time_per_char"].mean()
    avg_length = data["text_length"].mean()
    std_avg_time = data["avg_time_per_char"].std()
    data = data[(data["avg_time_per_char"] < avg_time + 3 * std_avg_time) | (data["text_length"] < avg_length / 3)]
    print("Excluded", length_old - len(data), "files with too slow speaking speed")

    # Print statistics again
    print_statistics(data)

    # Drop temporary columns again
    data = data.drop(columns=['duration', 'text_length', 'avg_time_per_char'])

    return data


# ======================================================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine prepared datasets')
    parser.add_argument('input_csv_path', type=str)
    parser.add_argument('output_csv_path', type=str)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--exclude', action='store_true')
    args = parser.parse_args()

    if (not (args.shuffle or args.replace or args.clean or args.exclude)):
        print("No operation given")
        exit()

    # Keep the german 0 as "null" string
    data = pd.read_csv(args.input_csv_path, keep_default_na=False)
    length_start = len(data)

    if (args.exclude):
        length_old = len(data)
        data = data[~data["wav_filename"].isin(excluded)]
        print("\nExcluded", length_old - len(data), "files which were marked for exclusion")

    if (args.shuffle):
        data = data.reindex(np.random.permutation(data.index))

    if (args.replace):
        data["transcript"] = data["transcript"].replace(replacer, regex=True)

    if (args.clean):
        data = clean(data)

    # Print summary
    length_diff = length_start - len(data)
    print("Excluded in total %i of %i files, those are %.1f%% of all files\n" % (
        length_diff, length_start, length_diff / length_start * 100))

    data.to_csv(args.output_csv_path, index=False, encoding='utf-8-sig')
