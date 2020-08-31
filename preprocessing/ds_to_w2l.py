import argparse
import hashlib
import os

import pandas as pd

# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert DeepSpeech datafiles for wav2letter"
    )
    parser.add_argument("input_csv_file", type=str)
    parser.add_argument("target_file", type=str)
    args = parser.parse_args()

    if os.path.exists(args.target_file):
        os.remove(args.target_file)

    dir_path = os.path.dirname(args.target_file)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    data = pd.read_csv(args.input_csv_file, keep_default_na=False)
    data["id"] = data["wav_filename"].apply(
        lambda x: hashlib.sha256(x.encode("utf-8")).hexdigest()
    )

    data.to_csv(
        args.target_file,
        index=False,
        encoding="utf-8",
        sep="\t",
        header=False,
        columns=["id", "wav_filename", "wav_filesize", "transcript"],
    )


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
