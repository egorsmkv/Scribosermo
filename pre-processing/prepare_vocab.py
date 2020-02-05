import argparse

import text_cleaning
import tqdm
from audiomate.utils import textfile


# ======================================================================================================================

def read_training_transcripts(path):
    transcripts = []
    lines = textfile.read_separated_lines_generator(path, separator=',', max_columns=3,
                                                    ignore_lines_starting_with=['wav_filename'])

    for entry in tqdm.tqdm(list(lines)):
        transcripts.append(entry[2])

    return transcripts


# ======================================================================================================================

parser = argparse.ArgumentParser(description='Clean text corpus.')
parser.add_argument('source_path', type=str)
parser.add_argument('target_path', type=str)
parser.add_argument('--training_csv', type=str)
args = parser.parse_args()

cleaned_sentences = []

# Load text corpus
with open(args.source_path, 'r', encoding="utf-8") as source_file:
    text_data = source_file.readlines()
    text_data = [x.strip() for x in text_data]

print("Adding sentences from source file")
for line in tqdm.tqdm(text_data):
    cleaned_sentence = text_cleaning.clean_sentence(line)
    cleaned_sentences.append(cleaned_sentence)

if args.training_csv is not None:
    print("Adding transcripts from training data")
    training_transcripts = read_training_transcripts(args.training_csv)
    cleaned_sentences.extend(training_transcripts)

# Save data
with open(args.target_path, 'a', encoding="utf-8") as target_file:
    text = '\n'.join(cleaned_sentences)
    target_file.write(text)
