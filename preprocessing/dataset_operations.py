import argparse
import json
import os
import time

import librosa
import numpy as np
import pandas as pd
import text_cleaning
from pandarallel import pandarallel


# ==================================================================================================


def get_duration(filename):
    """ Get duration of the wav file """
    length = librosa.get_duration(filename=filename)
    return length


# ==================================================================================================


def seconds_to_hours(secs):
    secs = int(secs)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    t = "{:d}:{:02d}:{:02d}".format(h, m, s)
    return t


# ==================================================================================================


def add_statistics_columns(data):
    """ Create some temporary columns """

    data["duration"] = data.parallel_apply(
        lambda x: get_duration(x.wav_filename), axis=1
    )
    data["text_length"] = data.parallel_apply(lambda x: len(x.transcript), axis=1)
    data["avg_time_per_char"] = data["duration"] / data["text_length"]

    return data


# ==================================================================================================


def print_statistics(data):
    print("\nStatistics:")

    col_data = data["duration"]
    msg = "Duration --- AVG: {:.2f}   STD: {:.2f}   MIN: {:.2f}   MAX: {:.2f}"
    print(msg.format(col_data.mean(), col_data.std(), col_data.min(), col_data.max()))

    col_data = data["text_length"]
    msg = "Text length --- AVG: {:.2f}   STD: {:.2f}   MIN: {:.2f}   MAX: {:.2f}"
    print(msg.format(col_data.mean(), col_data.std(), col_data.min(), col_data.max()))

    col_data = data["avg_time_per_char"]
    msg = "Time per char --- AVG: {:.2f}   STD: {:.2f}   MIN: {:.2f}   MAX: {:.2f}"
    print(msg.format(col_data.mean(), col_data.std(), col_data.min(), col_data.max()))

    print("")


# ==================================================================================================


def clean(data):
    # Keep only files longer than 1 second
    length_old = len(data)
    data = data[data["duration"] > 1]
    print("Excluded", length_old - len(data), "files with too short duration")

    # Keep only files less than 45 seconds
    length_old = len(data)
    data = data[data["duration"] < 45]
    print("Excluded", length_old - len(data), "files with too long duration")

    # Drop sentences with empty transcription
    length_old = len(data)
    data = data[data["text_length"] > 0]
    print("Excluded", length_old - len(data), "files with empty transcriptions")

    # Drop files spoken to fast
    length_old = len(data)
    avg_time = data["avg_time_per_char"].mean()
    data = data[data["avg_time_per_char"] > avg_time / 3]
    print("Excluded", length_old - len(data), "files with too fast char rate")

    # Drop files with a char rate below a second
    length_old = len(data)
    data = data[data["avg_time_per_char"] < 1]
    print("Excluded", length_old - len(data), "files with a much too slow char rate")

    # Keep only files which are not to slowly spoken,
    # except for very short files which may have longer pauses
    length_old = len(data)
    avg_time = data["avg_time_per_char"].mean()
    avg_length = data["text_length"].mean()
    std_avg_time = data["avg_time_per_char"].std()
    data = data[
        (data["avg_time_per_char"] < avg_time + 3 * std_avg_time)
        | (data["text_length"] < avg_length / 3)
    ]
    print("Excluded", length_old - len(data), "files with too slow speaking speed")

    return data


# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Clean and shuffle datasets")
    parser.add_argument("input_csv_path", type=str)
    parser.add_argument("output_csv_path", type=str)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--sort", action="store_true", help="Sort dataset by filesize")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--exclude", action="store_true")
    parser.add_argument("--nostats", action="store_true")
    args = parser.parse_args()

    pandarallel.initialize()
    print("This may take a few minutes ... ")

    file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    with open(file_path + "../data/excluded_files.json") as json_file:
        excluded = json.load(json_file)

    if not (args.shuffle or args.replace or args.clean or args.exclude):
        print("No operation given")
        exit()

    start_time = time.time()

    # Keep the german 0 as "null" string
    data = pd.read_csv(args.input_csv_path, keep_default_na=False)

    if not args.nostats:
        # Add statistics columns, save start size and duration and print data statistics
        data = add_statistics_columns(data)
        size_start = len(data)
        duration_start = data["duration"].sum()
        print_statistics(data)

    if args.exclude:
        length_old = len(data)
        data = data[~data["wav_filename"].isin(excluded)]
        msg = "Excluded {} files which were marked for exclusion"
        print(msg.format(length_old - len(data)))

    if args.shuffle:
        data = data.reindex(np.random.permutation(data.index))

    if args.sort:
        data = data.sort_values("wav_filesize")
        data = data.reset_index(drop=True)

    if args.replace:
        data["transcript"] = data["transcript"].str.lower()
        data["transcript"] = data["transcript"].parallel_apply(
            lambda x: text_cleaning.clean_sentence(x)[0]
        )

    if args.clean and not args.nostats:
        data = clean(data)

    if not args.nostats:
        # Print statistics again, save end size and duration and drop temporary columns
        size_end = len(data)
        time_end = data["duration"].sum()
        size_diff = size_start - size_end
        time_diff = duration_start - time_end
        print_statistics(data)
        data = data.drop(columns=["duration", "text_length", "avg_time_per_char"])

        # Print summary
        msg = "Excluded in total {} of {} files, those are {:.1f}% of all files"
        print(msg.format(size_diff, size_start, size_diff / size_start * 100))

        msg = "This are {} of {} hours, those are  {:.1f}% of the full duration"
        msg = msg.format(
            seconds_to_hours(time_diff),
            seconds_to_hours(duration_start),
            time_diff / duration_start * 100,
        )
        print(msg)

        msg = "Your dataset now has {} files and a duration of {} hours\n"
        print(msg.format(size_end, seconds_to_hours(time_end)))

    data.to_csv(args.output_csv_path, index=False, encoding="utf-8")
    end_time = time.time()
    msg = "Preparation took {} hours\n"
    print(msg.format(seconds_to_hours(end_time - start_time)))


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
