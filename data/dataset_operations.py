#! /usr/bin/env python

import argparse
import json
import os

import librosa
import numpy as np
import pandas as pd

# ======================================================================================================================

replacer = {
    "ä": "ae",
    "ö": "oe",
    "ü": "ue",

    "eins punkt null null null punkt null null null punkt null null null": "eine milliarde",
    "punkt null null null punkt null null null punkt null null null": "milliarden",
    "eins punkt null null null punkt null null null": "eine million",
    "punkt null null null punkt null null null": "millionen",
    "eins punkt null null null": "ein tausend",
    "punkt null null null": "tausend",
    "punkt null": ""
}

file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
with open(file_path + "excluded_files.json") as json_file:
    excluded = json.load(json_file)


# ======================================================================================================================

def get_duration(filename):
    """ Get duration of the wav file """
    length = librosa.get_duration(filename=filename)
    return length


# ======================================================================================================================

def seconds_to_hours(secs):
    secs = int(secs)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    t = '{:d}:{:02d}:{:02d}'.format(h, m, s)
    return t


# ======================================================================================================================

def add_statistics_columns(data):
    """ Create some temporary columns """

    data["duration"] = data.apply(lambda x: get_duration(x.wav_filename), axis=1)
    data["text_length"] = data.apply(lambda x: len(x.transcript), axis=1)
    data["avg_time_per_char"] = data["duration"] / data["text_length"]

    return data


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
    parser.add_argument('--nostats', action='store_true')
    args = parser.parse_args()

    if (not (args.shuffle or args.replace or args.clean or args.exclude)):
        print("No operation given")
        exit()

    # Keep the german 0 as "null" string
    data = pd.read_csv(args.input_csv_path, keep_default_na=False)

    if (not args.nostats):
        # Add statistics columns, save start size and duration and print data statistics
        data = add_statistics_columns(data)
        size_start = len(data)
        duration_start = data["duration"].sum()
        print_statistics(data)

    if (args.exclude):
        length_old = len(data)
        data = data[~data["wav_filename"].isin(excluded)]
        print("Excluded", length_old - len(data), "files which were marked for exclusion")

    if (args.shuffle):
        data = data.reindex(np.random.permutation(data.index))

    if (args.replace):
        data["transcript"] = data["transcript"].replace(replacer, regex=True)

    if (args.clean and not args.nostats):
        data = clean(data)

    if (not args.nostats):
        # Print statistics again, save end size and duration and drop temporary columns
        size_end = len(data)
        time_end = data["duration"].sum()
        size_diff = size_start - size_end
        time_diff = duration_start - time_end
        print_statistics(data)
        data = data.drop(columns=['duration', 'text_length', 'avg_time_per_char'])

        # Print summary
        print("Excluded in total %i of %i files, those are %.1f%% of all files" % (
            size_diff, size_start, size_diff / size_start * 100))
        print("This are %s of %s hours, those are %.1f%% of the full duration" % (
            seconds_to_hours(time_diff), seconds_to_hours(duration_start), time_diff / duration_start * 100))
        print("Your dataset now has {} files and a duration of {} hours\n".format(size_end, seconds_to_hours(time_end)))

    data.to_csv(args.output_csv_path, index=False, encoding='utf-8-sig')
