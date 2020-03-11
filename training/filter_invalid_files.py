import json
import os
import subprocess
import time

# ======================================================================================================================

checkpoint_path = "checkpoints/tvsmc/"
pretrained_checkpoint = "checkpoints/deepspeech-0.6.0-checkpoint/"
data_train_path = "data_prepared/tuda-voxforge-swc-mailabs-common_voice/train_azcem.csv"
data_dev_path = "data_prepared/tuda-voxforge-swc-mailabs-common_voice/dev_azcem.csv"
data_test_path = "data_prepared/tuda-voxforge-swc-mailabs-common_voice/test_azcem.csv"

exluded_path = os.path.dirname(os.path.realpath(__file__)) + "/../data/excluded_files.json"


# ======================================================================================================================

def add_files_to_excluded(problem_files):
    with open(exluded_path) as json_file:
        excluded = json.load(json_file)
    excluded.extend(problem_files)
    with open(exluded_path, "w+") as json_file:
        json.dump(excluded, json_file, indent=2)


# ======================================================================================================================

def main():
    while True:
        # Exclude files
        cmd = "python3 deepspeech-german/data/dataset_operations.py " + data_train_path + " " + data_train_path + \
              " --shuffle --exclude --nostats"

        sp = subprocess.Popen(['/bin/bash', '-c', cmd])
        sp.wait()

        # Run training
        check_path = checkpoint_path + str(int(time.time())) + "/"
        cmd = "/bin/bash deepspeech-german/training/train.sh " + check_path + " " + \
              data_train_path + " " + data_dev_path + " " + data_test_path + " 0 " + pretrained_checkpoint

        sp = subprocess.Popen(['/bin/bash', '-c', cmd])
        sp.wait()


# ======================================================================================================================

if __name__ == '__main__':
    main()
