import argparse
import itertools
import os

import pandas as pd


# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Create lexicon for wav2letter")
    parser.add_argument("train_csv_file", type=str)
    parser.add_argument("dev_csv_file", type=str)
    parser.add_argument("target_file", type=str)
    args = parser.parse_args()

    if os.path.exists(args.target_file):
        os.remove(args.target_file)

    dir_path = os.path.dirname(args.target_file)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    data1 = pd.read_csv(args.train_csv_file, keep_default_na=False)
    data1 = data1["transcript"].tolist()
    data2 = pd.read_csv(args.dev_csv_file, keep_default_na=False)
    data2 = data2["transcript"].tolist()

    data_all = data1
    data_all.extend(data2)

    words = [d.split() for d in data_all]
    words = list(itertools.chain.from_iterable(words))
    words = [w.strip() for w in words]
    words = [w for w in words if w != ""]
    words = set(words)

    lexicon = []
    for word in words:
        word = word + "\t" + " ".join(word) + " |"
        lexicon.append(word)

    lexicon = "\n".join(lexicon)
    with open(args.target_file, "w+", encoding="utf-8") as target_file:
        target_file.write(lexicon)


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
