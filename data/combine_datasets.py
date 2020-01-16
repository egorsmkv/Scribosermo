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

    args = parser.parse_args()

    tuda_path = args.tuda
    voxforge_path = args.voxforge
    swc_path = args.swc
    mailabs_path = args.mailabs
    cv_path = args.common_voice

    datasets = []

    if tuda_path:
        datasets.append("tuda")
    if voxforge_path:
        datasets.append("voxforge")
    if swc_path:
        datasets.append("swc")
    if mailabs_path:
        datasets.append("mailabs")
    if cv_path:
        datasets.append("common_voice")

    if (len(datasets) == 0):
        print("No datasets to combine")
        exit()

    data_train = []
    data_dev = []
    data_test = []
    for d in datasets:
        data_train.append(os.path.join(args.prepared_data_path, d, "train.csv"))
        data_dev.append(os.path.join(args.prepared_data_path, d, "dev.csv"))
        data_test.append(os.path.join(args.prepared_data_path, d, "test.csv"))

    # Combine all the files
    combined_csv_train = pd.concat([pd.read_csv(f, keep_default_na=False) for f in data_train])
    combined_csv_dev = pd.concat([pd.read_csv(f, keep_default_na=False) for f in data_dev])
    combined_csv_test = pd.concat([pd.read_csv(f, keep_default_na=False) for f in data_test])
    combined_csv_all = pd.concat([combined_csv_train, combined_csv_dev, combined_csv_test])

    # Export to csv again
    path = os.path.join(args.prepared_data_path, "-".join(datasets), "")
    if not os.path.exists(path):
        os.mkdir(path)
    combined_csv_train.to_csv(path + "train.csv", index=False, encoding='utf-8-sig')
    combined_csv_dev.to_csv(path + "dev.csv", index=False, encoding='utf-8-sig')
    combined_csv_test.to_csv(path + "test.csv", index=False, encoding='utf-8-sig')
    combined_csv_all.to_csv(path + "all.csv", index=False, encoding='utf-8-sig')
