import argparse

import pandas as pd


# ==================================================================================================
def extract(input_csv: str, output_txt: str):
    ds_pd = pd.read_csv(input_csv, encoding="utf-8", sep="\t")
    ds_tx = ds_pd["text"].tolist()
    text = "\n".join(ds_tx)

    with open(output_txt, "w+", encoding="utf-8") as file:
        file.write(text)


# ==================================================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Tool to extract sentences from the training data."
    )
    parser.add_argument(
        "--input_csv", help="Path to input dataset", type=str, required=True
    )
    parser.add_argument(
        "--output_txt", help="Path for the output", type=str, required=True
    )
    args = parser.parse_args()

    extract(args.input_csv, args.output_txt)


# ==================================================================================================

if __name__ == "__main__":
    main()
