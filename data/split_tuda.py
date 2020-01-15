#! /usr/bin/env python

import argparse
import os

import pandas as pd

# ======================================================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine prepared datasets')
    parser.add_argument('tuda_path', type=str)
    parser.add_argument('file_appendix', type=str)
    args = parser.parse_args()

    # Keep the german 0 as "null" string
    data = pd.read_csv(os.path.join(args.tuda_path, "all.csv"), keep_default_na=False)

    data_train = data[data["wav_filename"].str.contains("/train/")]
    data_dev = data[data["wav_filename"].str.contains("/dev/")]
    data_test = data[data["wav_filename"].str.contains("/test/")]

    print("Length of train, dev and test files:", len(data_train), len(data_dev), len(data_test))

    data_train.to_csv(os.path.join(args.tuda_path, "train" + args.file_appendix + ".csv"), index=False,
                      encoding='utf-8-sig')
    data_dev.to_csv(os.path.join(args.tuda_path, "dev" + args.file_appendix + ".csv"), index=False,
                    encoding='utf-8-sig')
    data_test.to_csv(os.path.join(args.tuda_path, "test" + args.file_appendix + ".csv"), index=False,
                     encoding='utf-8-sig')
