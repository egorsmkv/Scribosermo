import argparse
import os

import text_cleaning
import tqdm
from audiomate.utils import textfile


# ==================================================================================================


def read_training_transcripts(path):
    transcripts = []
    lines = textfile.read_separated_lines_generator(
        path, separator=",", max_columns=3, ignore_lines_starting_with=["wav_filename"]
    )

    for entry in tqdm.tqdm(list(lines)):
        transcripts.append(entry[2])

    return transcripts


# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Clean text corpus.")
    parser.add_argument("source_path", type=str)
    parser.add_argument("target_path", type=str)
    parser.add_argument("--training_csv", type=str)
    parser.add_argument("--replace_umlauts", action="store_true")
    args = parser.parse_args()

    all_cleaned_sentences = []
    text_data = []

    # Load text corpora
    for file in os.listdir(args.source_path):
        if file.endswith(".txt") or file.endswith(".de"):
            path = os.path.join(args.source_path, file)
            with open(path, "r", encoding="utf-8") as source_file:
                content = source_file.readlines()
                content = [x.strip() for x in content]
                text_data.extend(content)

    print("Adding sentences from source file")
    print(len(text_data))
    csl = text_cleaning.clean_sentence_list(text_data, args.replace_umlauts)
    all_cleaned_sentences.extend(csl)

    if args.training_csv is not None:
        print("Adding transcripts from training data")
        training_transcripts = read_training_transcripts(args.training_csv)
        csl = text_cleaning.clean_sentence_list(training_transcripts, args)
        all_cleaned_sentences.extend(csl)

    # Save data
    with open(args.target_path, "w+", encoding="utf-8") as target_file:
        text = "\n".join(all_cleaned_sentences)
        target_file.write(text)


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
