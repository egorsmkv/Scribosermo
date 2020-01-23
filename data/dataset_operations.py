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
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-14-04_Realtek.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-13-08_Yamaha.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-39-55_Samson.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-13-41_Realtek.wav"
    "/DeepSpeech/data_original/tuda/train/2014-03-17-13-10-00_Realtek.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-39-46_Kinect-RAW.wav",
    "/DeepSpeech/data_original/tuda/train/2014-03-17-13-07-00_Yamaha.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-13-41_Kinect-Beam.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-04-54_Yamaha.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-14-04_Samson.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-11-49_Kinect-Beam.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-39-46_Samson.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-22-25_Kinect-RAW.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-13-17_Samson.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-11-22_Realtek.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-22-29_Yamaha.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-39-55_Yamaha.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-16-32_Kinect-RAW.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-15-15_Kinect-RAW.wav",
    "/DeepSpeech/data_original/tuda/train/2014-03-17-13-10-00_Realtek.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-14-45_Samson.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-14-23_Yamaha.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-15-15_Realtek.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-22-45_Kinect-RAW.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-22-45_Samson.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-14-04_Yamaha.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-22-45_Realtek.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-39-55_Kinect-RAW.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-09-17_Realtek.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-16-32_Realtek.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-14-04_Kinect-Beam.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-14-23_Kinect-Beam.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-11-49_Yamaha.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-11-04_Kinect-RAW.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-15-15_Kinect-Beam.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-04-54_Realtek.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-14-23_Samson.wav",
    "/DeepSpeech/data_original/tuda/train/2014-03-17-13-10-00_Microsoft-Kinect-Raw.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-11-49_Realtek.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-22-21_Kinect-RAW.wav",
    "/DeepSpeech/data_original/tuda/train/2014-08-04-13-13-17_Kinect-Beam.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000106_0000000752_1689330_1690410.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000184_0000000302_955470_956630.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000172_0000000282_1657790_1659180.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000277_0000000482_1202110_1203510.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000009_0000000314_81760_82900.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000100_0000000771_837710_838880.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000363_0000000663_1277540_1278560.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000017_0000000146_263780_264860.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000267_0000000896_270170_271280.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000134_0000000196_577100_578230.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000267_0000000896_281560_282690.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000571_0000001131_748930_750330.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000100_0000001123_144630_146080.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000057_0000000744_1079140_1081240.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000106_0000000870_2555750_2557120.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000434_0000000919_1110120_1111530.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000520_0000001011_2026030_2027810.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000007_0000001004_230570_231660.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000394_0000000721_162470_163880.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000408_0000000748_6317540_6318960.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000063_0000000096_1206320_1207890.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000223_0000000383_469800_471430.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000434_0000000909_953290_955180.wav",
    "/DeepSpeech/data_original/tuda/train/2014-03-17-13-15-57_Kinect-Beam.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000508_0000000979_1129450_1130560.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000007_0000000918_2934060_2935190.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000497_0000000951_70300_71500.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000169_0000000265_1077750_1078970.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000348_0000000632_7514140_7515600.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000348_0000000627_901300_903100.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000277_0000000610_1023840_1025710.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000163_0000000253_52060_55720.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000401_0000000732_184000_185280.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000481_0000000926_900590_901940.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000117_0000000171_190510_192430.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000434_0000000919_344700_346650.wav",
    "/DeepSpeech/data_prepared/swc/audio/00000402_0000000736_2943620_2944990.wav",
    "/DeepSpeech/data_original/mailabs/by_book/by_book/female/angela_merkel/merkel_alone/wavs/Kanzlerin_42_13_f000037.wav"
]


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
    args = parser.parse_args()

    if (not (args.shuffle or args.replace or args.clean or args.exclude)):
        print("No operation given")
        exit()

    # Keep the german 0 as "null" string
    data = pd.read_csv(args.input_csv_path, keep_default_na=False)

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

    if (args.clean):
        data = clean(data)

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
