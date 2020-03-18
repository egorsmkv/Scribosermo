import argparse
import re
from functools import partial
from multiprocessing import Pool

import text_cleaning
import tqdm
from audiomate.utils import textfile

# ======================================================================================================================

umlaut_replacers = {
    "ä": "ae",
    "ö": "oe",
    "ü": "ue",
    "ß": "ss"
}
umlaut_pattern = re.compile('|'.join(umlaut_replacers.keys()))


# ======================================================================================================================

def read_training_transcripts(path):
    transcripts = []
    lines = textfile.read_separated_lines_generator(path, separator=',', max_columns=3,
                                                    ignore_lines_starting_with=['wav_filename'])

    for entry in tqdm.tqdm(list(lines)):
        transcripts.append(entry[2])

    return transcripts


# ======================================================================================================================

def clean_line(line, args):
    line = text_cleaning.clean_sentence(line)
    if (args.replace_umlauts):
        line = umlaut_pattern.sub(lambda x: umlaut_replacers[x.group()], line)
    return line


# ======================================================================================================================

def clean_sentence_list(sentences, args):
    cl_func = partial(clean_line, args=args)

    with Pool() as p:
        cleaned_sentences = list(tqdm.tqdm(p.imap(cl_func, sentences), total=len(sentences)))

    return cleaned_sentences


# ======================================================================================================================

def main():
    parser = argparse.ArgumentParser(description='Clean text corpus.')
    parser.add_argument('source_path', type=str)
    parser.add_argument('target_path', type=str)
    parser.add_argument('--training_csv', type=str)
    parser.add_argument('--replace_umlauts', action='store_true')
    args = parser.parse_args()

    all_cleaned_sentences = []

    # Load text corpus
    with open(args.source_path, 'r', encoding="utf-8") as source_file:
        text_data = source_file.readlines()
        text_data = [x.strip() for x in text_data]

    print("Adding sentences from source file")
    cleaned_sentences = clean_sentence_list(text_data, args)
    all_cleaned_sentences.extend(cleaned_sentences)

    if args.training_csv is not None:
        print("Adding transcripts from training data")
        training_transcripts = read_training_transcripts(args.training_csv)
        cleaned_sentences = clean_sentence_list(training_transcripts, args)
        all_cleaned_sentences.extend(cleaned_sentences)

    # Save data
    with open(args.target_path, 'w+', encoding="utf-8") as target_file:
        text = '\n'.join(all_cleaned_sentences)
        target_file.write(text)


# ======================================================================================================================

if __name__ == '__main__':
    main()
