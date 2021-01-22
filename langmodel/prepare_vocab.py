import argparse
import os
import sys

# The if block is required for isort
if True:  # pylint: disable=using-constant-test
    file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    sys.path.insert(1, file_path + "../preprocessing")
    import text_cleaning


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
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    txt_files = os.listdir(args.input_dir)
    txt_files = [f for f in txt_files if f.endswith(".txt")]
    print("Found {} text files\n".format(len(txt_files)))

    total_lines = 0
    for tfile in txt_files:
        print("Cleaning: {} ...".format(tfile))

        read_path = os.path.join(args.input_dir, tfile)
        with open(read_path, "r", encoding="utf-8") as source_file:
            sentences = source_file.readlines()

        csl = text_cleaning.clean_sentence_list(sentences)
        text = "\n".join(csl) + "\n"
        total_lines += len(csl)

        save_path = os.path.join(args.output_dir, tfile)
        with open(save_path, "w+", encoding="utf-8") as file:
            file.write(text)

    print("Processed {} sentences".format(total_lines))


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
