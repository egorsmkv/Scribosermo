import argparse
import os

import tqdm
from audiomate.utils import textfile

import text_cleaning

# ==================================================================================================


def read_training_transcripts(path):
    transcripts = []
    lines = textfile.read_separated_lines_generator(
        path, separator=",", max_columns=3, ignore_lines_starting_with=["wav_filename"]
    )

    for entry in list(lines):
        transcripts.append(entry[2])

    return transcripts


# ==================================================================================================


def handle_file_content(sentences, save_path):
    """ Normalize list of sentences and append them to the output file """

    csl = text_cleaning.clean_sentence_list(sentences)
    text = "\n".join(csl) + "\n"

    with open(save_path, "a+", encoding="utf-8") as file:
        file.write(text)


# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Clean text corpus.")
    parser.add_argument("target_path", type=str)
    parser.add_argument("--source_dir", type=str)
    parser.add_argument("--training_csv", type=str)
    args = parser.parse_args()

    if os.path.exists(args.target_path):
        os.remove(args.target_path)

    dir_path = os.path.dirname(args.target_path)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    if args.source_dir is not None:
        print("Loading text corpora ...")

        files = os.listdir(args.source_dir)
        files = [f for f in files if not (f.endswith(".gz") or f.endswith(".tgz"))]
        for file in tqdm.tqdm(files):
            path = os.path.join(args.source_dir, file)
            with open(path, "r", encoding="utf-8") as source_file:
                content = source_file.readlines()
                content = [x.strip() for x in content]

            # Clean and save file after file to reduce memory demands
            msg = "Adding {} sentences from file: {}"
            print(msg.format(len(content), file))
            handle_file_content(content, args.target_path)

    if args.training_csv is not None:
        print("Loading training transcripts ...")

        content = read_training_transcripts(args.training_csv)
        print("Adding {} transcripts from training data".format(len(content)))
        handle_file_content(content, args.target_path)


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
