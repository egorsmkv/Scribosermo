#! /usr/bin/env python

import argparse

import numpy as np
import pandas as pd

replacer = {
    "ä": "ae",
    "ö": "oe",
    "ü": "ue"
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine prepared datasets')
    parser.add_argument('input_csv_path', type=str)
    parser.add_argument('output_csv_path', type=str)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()

    if (not (args.shuffle or args.replace)):
        print("No operation given")
        exit()

    data = pd.read_csv(args.input_csv_path)

    if (args.shuffle):
        data = data.reindex(np.random.permutation(data.index))

    if (args.replace):
        data["transcript"] = data["transcript"].replace(replacer, regex=True)

        data.to_csv(args.output_csv_path, index=False, encoding='utf-8-sig')
