import json
import os
import subprocess

# ======================================================================================================================

checkpoint_path = "checkpoints/tfsmc/"
data_train_path = "data_prepared/tuda-voxforge-swc-mailabs-common_voice/all_azce.csv"
data_dev_path = "data_prepared/voxforge/dev_azce.csv"
data_test_path = "data_prepared/voxforge/test_azce.csv"

exluded_path = os.path.dirname(os.path.realpath(__file__)) + "/../data/excluded_files.json"


# ======================================================================================================================

def add_files_to_excluded(problem_files):
    with open(exluded_path) as json_file:
        excluded = json.load(json_file)
    excluded.extend(problem_files)
    with open(exluded_path, "w+") as json_file:
        json.dump(excluded, json_file, indent=4)


# ======================================================================================================================

def main():
    while True:
        # Exclude files
        cmd = "python3 deepspeech-german/data/dataset_operations.py " + data_train_path + " " + data_train_path + \
              " --shuffle --exclude --nostats"

        sp = subprocess.Popen(['/bin/bash', '-c', cmd])
        sp.wait()

        # Run training
        cmd = "/bin/bash deepspeech-german/training/train.sh " + checkpoint_path + " " + \
              data_train_path + " " + data_dev_path + " " + data_test_path + " 0 1"

        sp = subprocess.Popen(['/bin/bash', '-c', cmd])
        sp.wait()


# ======================================================================================================================

if __name__ == '__main__':
    main()
