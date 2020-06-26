import argparse
import os

import pandas as pd


# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Split dataset into parts")
    parser.add_argument("csv_to_split_path", type=str)
    parser.add_argument("--file_appendix", type=str, default="")
    parser.add_argument(
        "--tuda", action="store_true", help="Split into correct tuda parts"
    )
    parser.add_argument(
        "--common_voice",
        action="store_true",
        help="Remove test samples where transcript also in trainset",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="",
        help="Split into parts. Argument usage: --split '70|15|15'",
    )
    args = parser.parse_args()

    if args.tuda or args.split != "":
        # Keep the german 0 as "null" string
        data = pd.read_csv(args.csv_to_split_path, keep_default_na=False)

    if args.tuda:
        data_train = data[data["wav_filename"].str.contains("/train/")]
        data_dev = data[data["wav_filename"].str.contains("/dev/")]
        data_test = data[data["wav_filename"].str.contains("/test/")]

    elif args.common_voice:
        data_test = pd.read_csv(args.csv_to_split_path, keep_default_na=False)
        path = args.csv_to_split_path.replace("test", "dev")
        data_dev = pd.read_csv(path, keep_default_na=False)
        path = args.csv_to_split_path.replace("test", "train")
        data_train = pd.read_csv(path, keep_default_na=False)

        # Move samples with same transcription from devset to trainset
        overlap_td = set(data_train["transcript"]).intersection(data_dev["transcript"])
        csvs = [data_train, data_dev[data_dev["transcript"].isin(overlap_td)]]
        data_train = pd.concat(csvs)
        data_dev = data_dev[~data_dev["transcript"].isin(overlap_td)]

        # Move samples with same transcription from testset to trainset
        overlap_tt = set(data_train["transcript"]).intersection(data_test["transcript"])
        csvs = [data_train, data_test[data_test["transcript"].isin(overlap_tt)]]
        data_train = pd.concat(csvs)
        data_test = data_test[~data_test["transcript"].isin(overlap_tt)]

    elif args.split != "":
        splits = [int(s) for s in args.split.split("|")]
        if len(splits) != 3 or sum(splits) != 100:
            raise ValueError

        splits = [s * 0.01 for s in splits]
        datasize = len(data)

        data_train = data.iloc[: int(datasize * splits[0])]
        data_dev = data.iloc[
            int(datasize * splits[0]) : int(datasize * splits[0] + datasize * splits[1])
        ]
        data_test = data.iloc[int(datasize * splits[0] + datasize * splits[1]) :]

    msg = "Length of train, dev and test files: {} {} {}"
    print(msg.format(len(data_train), len(data_dev), len(data_test),))

    outpath = os.path.dirname(args.csv_to_split_path)
    data_train.to_csv(
        os.path.join(outpath, "train" + args.file_appendix + ".csv"),
        index=False,
        encoding="utf-8",
    )
    data_dev.to_csv(
        os.path.join(outpath, "dev" + args.file_appendix + ".csv"),
        index=False,
        encoding="utf-8",
    )
    data_test.to_csv(
        os.path.join(outpath, "test" + args.file_appendix + ".csv"),
        index=False,
        encoding="utf-8",
    )


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
