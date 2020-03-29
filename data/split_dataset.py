#! /usr/bin/env python

import argparse
import os

import pandas as pd

# ======================================================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine prepared datasets')
    parser.add_argument('all_csv_path', type=str)
    parser.add_argument('--file_appendix', type=str, default="")
    parser.add_argument('--tuda', action='store_true', help="Split into correct tuda parts")
    parser.add_argument('--split', type=str, default="", help="Split into parts. Argument usage: --split '70|15|15'")
    args = parser.parse_args()

    # Keep the german 0 as "null" string
    data = pd.read_csv(os.path.join(args.all_csv_path), keep_default_na=False)

    if (args.tuda):
        data_train = data[data["wav_filename"].str.contains("/train/")]
        data_dev = data[data["wav_filename"].str.contains("/dev/")]
        data_test = data[data["wav_filename"].str.contains("/test/")]

    elif (args.split != ""):
        splits = [int(s) for s in args.split.split("|")]
        if (len(splits) != 3 or sum(splits) != 100):
            raise ValueError

        splits = [s * 0.01 for s in splits]
        datasize = len(data)

        data_train = data.iloc[:int(datasize * splits[0])]
        data_dev = data.iloc[int(datasize * splits[0]):int(datasize * splits[0] + datasize * splits[1])]
        data_test = data.iloc[int(datasize * splits[0] + datasize * splits[1]):]

    print("Length of train, dev and test files:", len(data_train), len(data_dev), len(data_test))

    outpath = os.path.dirname(args.all_csv_path)
    data_train.to_csv(os.path.join(outpath, "train" + args.file_appendix + ".csv"), index=False, encoding='utf-8')
    data_dev.to_csv(os.path.join(outpath, "dev" + args.file_appendix + ".csv"), index=False, encoding='utf-8')
    data_test.to_csv(os.path.join(outpath, "test" + args.file_appendix + ".csv"), index=False, encoding='utf-8')
