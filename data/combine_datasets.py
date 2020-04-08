#! /usr/bin/env python

import argparse
import os

import pandas as pd

# ======================================================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine prepared datasets')
    parser.add_argument('prepared_data_path', type=str)
    parser.add_argument('--tuda', action='store_true')
    parser.add_argument('--voxforge', action='store_true')
    parser.add_argument('--swc', action='store_true')
    parser.add_argument('--mailabs', action='store_true')
    parser.add_argument('--common_voice', action='store_true')
    parser.add_argument('--google_wavenet', action='store_true')
    parser.add_argument('--data_appendix', type=str, default="")
    parser.add_argument('--files', type=str, default="", help="List of files, separated with a space")
    parser.add_argument('--files_output', type=str, default="", help="Output path for combined list of files")

    args = parser.parse_args()
    datasets = []

    if args.tuda:
        datasets.append("tuda")
    if args.voxforge:
        datasets.append("voxforge")
    if args.swc:
        datasets.append("swc")
    if args.mailabs:
        datasets.append("mailabs")
    if args.common_voice:
        datasets.append("common_voice")
    if args.google_wavenet:
        datasets.append("google_wavenet")

    if (len(datasets) > 0):
        data_train = []
        data_dev = []
        data_test = []
        for d in datasets:
            data_train.append(os.path.join(args.prepared_data_path, d, "train" + args.data_appendix + ".csv"))
            data_dev.append(os.path.join(args.prepared_data_path, d, "dev" + args.data_appendix + ".csv"))
            data_test.append(os.path.join(args.prepared_data_path, d, "test" + args.data_appendix + ".csv"))

        # Combine all the files
        combined_csv_train = pd.concat([pd.read_csv(f, keep_default_na=False) for f in data_train])
        combined_csv_dev = pd.concat([pd.read_csv(f, keep_default_na=False) for f in data_dev])
        combined_csv_test = pd.concat([pd.read_csv(f, keep_default_na=False) for f in data_test])
        combined_csv_all = pd.concat([combined_csv_train, combined_csv_dev, combined_csv_test])

        # Export to csv again
        path = os.path.join(args.prepared_data_path, "-".join(datasets), "")
        if not os.path.exists(path):
            os.mkdir(path)
        combined_csv_train.to_csv(path + "train" + args.data_appendix + ".csv", index=False, encoding='utf-8')
        combined_csv_dev.to_csv(path + "dev" + args.data_appendix + ".csv", index=False, encoding='utf-8')
        combined_csv_test.to_csv(path + "test" + args.data_appendix + ".csv", index=False, encoding='utf-8')
        combined_csv_all.to_csv(path + "all" + args.data_appendix + ".csv", index=False, encoding='utf-8')

    elif (args.files != "" and args.files_output != ""):
        files = args.files.split(" ")

        combined_csv = pd.concat([pd.read_csv(f, keep_default_na=False) for f in files])
        combined_csv.to_csv(args.files_output, index=False, encoding='utf-8')
