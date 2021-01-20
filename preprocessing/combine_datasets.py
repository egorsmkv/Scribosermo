import argparse
import os

import pandas as pd

# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Combine prepared datasets")
    parser.add_argument(
        "--files", type=str, default="", help="List of files, separated with space"
    )
    parser.add_argument(
        "--file_output", type=str, help="Output path for combined list of files"
    )
    args = parser.parse_args()

    dir_path = os.path.dirname(args.file_output)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    files = args.files.split(" ")
    combined_csv = pd.concat(
        [pd.read_csv(f, sep="\t", keep_default_na=False) for f in files]
    )
    combined_csv.to_csv(args.file_output, index=False, sep="\t", encoding="utf-8")


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
