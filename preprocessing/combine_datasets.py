import argparse
import os
import random
import wave
from functools import partial
from multiprocessing import Pool

import pandas as pd
import tensorflow as tf
import tqdm

# ==================================================================================================


def read_and_validate(cfiles):

    csvs = []
    print("Loading files ...")
    for cfile in tqdm.tqdm(cfiles):
        # Check all required columns are existing
        df = pd.read_csv(cfile, sep="\t", keep_default_na=False)
        if not {"duration", "filepath", "text"}.issubset(set(df.columns.values)):
            raise ValueError("A column is missing in: {}".format(cfile))
        csvs.append(df)

    print("Updating paths ...")
    for df, cfile in zip(csvs, cfiles):
        # Make paths absolute if they aren't already
        if not os.path.isabs(df["filepath"][0]):
            dir_path = os.path.dirname(cfile)

            def join_path(file, path: str = ""):
                return os.path.abspath(os.path.join(path, file))

            # Without using partial, pylint complains about an "cell-var-from-loop" error
            pfunc = partial(join_path, path=dir_path)
            df["filepath"] = df["filepath"].apply(pfunc)

    print("Checking paths ...")
    for df in tqdm.tqdm(csvs):
        # Check that every file exists
        fpaths = df["filepath"].tolist()

        with Pool() as p:
            existing = list(p.imap(os.path.exists, fpaths))

        if not all(existing):
            idx = existing.index(False)
            apath = fpaths[idx]
            print("ERROR: Can't find this file: {}".format(apath))

    print("Checking random audio samples ...")
    for df in tqdm.tqdm(csvs):
        # Check some files of each dataset to ensure they have the correct audio format
        fpaths = df["filepath"].tolist()
        fpaths = random.sample(fpaths, 3)

        for fpath in fpaths:
            with open(fpath, "rb") as afile:
                audio = wave.open(afile)
                nchannels = audio.getnchannels()
                swidth = audio.getsampwidth()
                frate = audio.getframerate()

                if nchannels != 1 or swidth != 2 or frate != 16000:
                    msg = (
                        "ERROR: Audio can only have 1 channel, a sample width of"
                        " 2 bytes and a frame rate of 16kHz. But following file has"
                        " {} channels, sample width of {} and frame rate of {}: {}"
                    )
                    print(msg.format(nchannels, swidth, frate, fpath))

            try:
                audio_binary = tf.io.read_file(fpath)
                _ = tf.audio.decode_wav(audio_binary)
            except tf.errors.InvalidArgumentError:
                print("ERROR: Couldn't read file: {}".format(fpath))

    return csvs


# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Combine prepared datasets")
    parser.add_argument(
        "--files_str", type=str, default="", help="List of files, separated with space"
    )
    parser.add_argument(
        "--files_txt",
        type=str,
        default="",
        help="Path to a list of files, separated by newline",
    )
    parser.add_argument(
        "--file_output", type=str, help="Output path for combined list of files"
    )
    args = parser.parse_args()

    if args.files_str == "" and args.files_txt == "":
        raise ValueError("No files given!")

    if args.files_str != "":
        cfiles = args.files_str.split(" ")
    else:
        with open(args.files_txt, "r", encoding="utf-8") as file:
            cfiles = file.readlines()
            cfiles = [f.strip() for f in cfiles]
            cfiles = [f for f in cfiles if f != ""]

    csvs = read_and_validate(cfiles)

    dir_path = os.path.dirname(args.file_output)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    print("Combining and writing file ...")
    combined_csv = pd.concat(csvs, axis=0, join="inner")
    combined_csv.to_csv(args.file_output, index=False, sep="\t", encoding="utf-8")


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
