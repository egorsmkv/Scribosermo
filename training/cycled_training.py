import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

# ==================================================================================================

steps_per_dataset = 5
start_with_checkpoint = "/DeepSpeech/checkpoints/deepspeech-0.6.0-checkpoint/"

# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Run cycled training")
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("prepared_data_path", type=str)
    parser.add_argument("data_appendix", type=str)
    parser.add_argument("--tuda", action="store_true")
    parser.add_argument("--voxforge", action="store_true")
    parser.add_argument("--mailabs", action="store_true")
    parser.add_argument("--swc", action="store_true")
    parser.add_argument("--common_voice", action="store_true")
    args = parser.parse_args()

    datasets = []
    if args.tuda:
        datasets.append("tuda")
    if args.voxforge:
        datasets.append("voxforge")
    if args.mailabs:
        datasets.append("mailabs")
    if args.swc:
        datasets.append("swc")
    if args.common_voice:
        datasets.append("common_voice")

    if len(datasets) == 0:
        print("No datasets to train with")
        sys.exit()

    data_train_all = None
    data_dev_all = None
    data_test_all = None
    data_path_cycled = os.path.join(args.prepared_data_path, "cycled_data")
    data_train_path_cycled = os.path.join(data_path_cycled, "train.csv")
    data_dev_path_cycled = os.path.join(data_path_cycled, "dev.csv")
    data_test_path_cycled = os.path.join(data_path_cycled, "test.csv")

    if os.path.isdir(data_path_cycled):
        shutil.rmtree(data_path_cycled)
    os.mkdir(data_path_cycled)

    # Create dev and test dataset
    for d in datasets:
        file = "dev" + args.data_appendix + ".csv"
        data_dev_path = os.path.join(args.prepared_data_path, d, file)
        data_dev = pd.read_csv(data_dev_path, keep_default_na=False)
        file = "test" + args.data_appendix + ".csv"
        data_test_path = os.path.join(args.prepared_data_path, d, file)
        data_test = pd.read_csv(data_test_path, keep_default_na=False)

        if data_dev_all is None:
            data_dev_all = data_dev
        else:
            data_dev_all = pd.concat([data_dev_all, data_dev])
        if data_test_all is None:
            data_test_all = data_test
        else:
            data_test_all = pd.concat([data_test_all, data_test])

    # Shuffle them
    data_dev_all = data_dev_all.iloc[np.random.permutation(len(data_dev_all))]
    data_test_all = data_test_all.iloc[np.random.permutation(len(data_test_all))]
    # And save them to a file
    data_dev_all.to_csv(data_dev_path_cycled, index=False, encoding="utf-8")
    data_test_all.to_csv(data_test_path_cycled, index=False, encoding="utf-8")

    # Run the cycled training
    use_checkpoint = start_with_checkpoint
    for j, d in enumerate(datasets):
        file = "train" + args.data_appendix + ".csv"
        data_train_path = os.path.join(args.prepared_data_path, d, file)
        data_train = pd.read_csv(data_train_path, keep_default_na=False)
        print("\nStarted to add a new dataset with {} entries".format(len(data_train)))

        # Sort the dataset before splitting and rebuild the index
        # that the training starts with the shorter samples (easier samples) first
        data_train = data_train.sort_values("wav_filesize")
        data_train = data_train.reset_index(drop=True)

        for i in range(steps_per_dataset):
            split_size = int(len(data_train) / steps_per_dataset)
            split_start = split_size * i
            # Don't miss out data at the end if split was not even
            if i != steps_per_dataset - 1:
                split_end = split_size * (i + 1)
            else:
                split_end = len(data_train)

            data_train_split = data_train[split_start:split_end]

            if data_train_all is None:
                data_train_all = data_train_split
            else:
                data_train_all = pd.concat([data_train_all, data_train_split])

            # Sort again and save as file
            data_train_all = data_train_all.sort_values("wav_filesize")
            data_train_all = data_train_all.reset_index(drop=True)
            data_train_all.to_csv(data_train_path_cycled, index=False, encoding="utf-8")

            cmd = "/bin/bash /DeepSpeech/deepspeech-polyglot/training/train.sh {} {} {} {} 0 {}"
            cmd = cmd.format(
                args.checkpoint_path,
                data_train_path_cycled,
                data_dev_path_cycled,
                data_test_path_cycled,
                use_checkpoint,
            )

            if use_checkpoint != "--":
                use_checkpoint = "--"

            loginfo = "\nDataset {}/{} - Cycle {}/{}"
            loginfo += "\nRunning next trainstep with {} training samples"
            loginfo = loginfo.format(
                j + 1, len(datasets), i, steps_per_dataset, len(data_train_all)
            )
            # Log the info and then run the command
            # Use echo instead of normal print to fix incorrect logging order with slurm execution
            cmd = "echo -e '{}' && ".format(loginfo) + cmd
            sp = subprocess.Popen(["/bin/bash", "-c", cmd])
            sp.wait()


# ==================================================================================================

if __name__ == "__main__":
    main()
