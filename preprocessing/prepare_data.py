"""
1. Load all corpora where a path is given.
2. Clean transcriptions.
3. Merge all corpora
4. Create Train/Dev/Test splits
5. Export for DeepSpeech
"""

import argparse
import json
import os
import re
import shutil

import audiomate
import pandas as pd
import text_cleaning
import tqdm
from audiomate.corpus import io, subset
from pydub import AudioSegment


# ==================================================================================================


def clean_transcriptions(corpus):
    print("Cleaning transcriptions ...")
    for utterance in tqdm.tqdm(corpus.utterances.values()):
        ll = utterance.label_lists[audiomate.corpus.LL_WORD_TRANSCRIPT]

        for label in ll:
            label.value = text_cleaning.clean_sentence(label.value)[0]


# ==================================================================================================


def write_dataset(dataset, path):
    """ Saves a dataset in DeepSpeech format.
    Input a list of dicts containing at least 'file' and 'transcription' keys. """
    print("Saving dataset ...")

    # Create or delete and recreate target directory
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    path_audios = os.path.join(path, "audios")
    os.makedirs(path_audios, exist_ok=True)

    new_dataset = []

    for entry in tqdm.tqdm(dataset):
        if not os.path.exists(entry["file"]):
            print("This file does not exist: {}".format(entry["file"]))
            continue

        name, extension = os.path.splitext(os.path.basename(entry["file"]))
        out_file = os.path.join(path_audios, name + ".wav")

        audio = AudioSegment.from_file(entry["file"], extension)
        audio.export(out_file, format="wav", bitrate="16k", parameters=["-ac", "1"])

        cleaned_transcript = text_cleaning.clean_sentence(entry["transcription"])[0]

        new_entry = {
            "wav_filename": out_file,
            "wav_filesize": os.path.getsize(out_file),
            "transcript": cleaned_transcript,
        }
        new_dataset.append(new_entry)

    dataset = pd.DataFrame(new_dataset)
    dataset.to_csv(
        os.path.join(path, "all.csv"), index=False, encoding="utf-8",
    )


# ==================================================================================================


def load_basformtask(path):
    print("Loading transcripts ...")
    dir_names = os.listdir(path)
    dir_names = [os.path.join(path, d) for d in dir_names]
    dir_names = [d for d in dir_names if os.path.isdir(d)]

    tags_pattern = re.compile("<.*?>")

    dataset = []
    for d in tqdm.tqdm(dir_names):
        files = os.listdir(d)
        files = [os.path.join(d, f) for f in files]

        annots = [f for f in files if f.endswith("_annot.json")]
        for annot in annots:
            wav = str(os.path.basename(annot).split("_")[0]) + ".wav"
            wav = os.path.join(d, wav)

            with open(annot, "r", encoding="utf-8") as file:
                annot = json.load(file)

            trans = annot["levels"][0]["items"][0]["labels"][1]["value"]
            trans = re.sub(tags_pattern, "", trans)

            entry = {
                "file": wav,
                "transcription": trans,
                "speaker": os.path.basename(d),
            }
            dataset.append(entry)

    # Drop file including signs for incomplete words or repetitions
    cleaned_set = []
    for entry in dataset:
        if not "~" in entry["transcription"] and not "*" in entry["transcription"]:
            cleaned_set.append(entry)
    msg = "Dropped {}/{} files with speech errors"
    print(msg.format(len(dataset) - len(cleaned_set), len(dataset)))

    return cleaned_set


# ==================================================================================================


def load_bassprecherinnen(path):
    print("Loading transcripts ...")
    dir_names = os.listdir(path)
    dir_names = [os.path.join(path, d) for d in dir_names]
    dir_names = [d for d in dir_names if os.path.isdir(d)]

    tags_pattern = re.compile("<.*?>")

    dataset = []
    for d in tqdm.tqdm(dir_names):
        files = os.listdir(d)
        files = [os.path.join(d, f) for f in files]

        annots = [f for f in files if f.endswith("_annot.json")]
        for annot in annots:
            wav = str(os.path.basename(annot).split("_")[0]) + ".wav"
            wav = os.path.join(d, wav)

            with open(annot, "r", encoding="utf-8") as file:
                annot = json.load(file)

            trans = []
            for lab in annot["levels"][1]["items"]:
                trans.append(lab["labels"][0]["value"])
            trans = " ".join(trans)
            trans = re.sub(tags_pattern, "", trans)

            entry = {
                "file": wav,
                "transcription": trans,
                "speaker": os.path.basename(d),
            }
            dataset.append(entry)

    # Drop file including signs for incomplete words or repetitions
    cleaned_set = []
    for entry in dataset:
        if not "~" in entry["transcription"] and not "*" in entry["transcription"]:
            cleaned_set.append(entry)
    msg = "Dropped {}/{} files with speech errors"
    print(msg.format(len(dataset) - len(cleaned_set), len(dataset)))

    return cleaned_set


# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training.")
    parser.add_argument("target_path", type=str)
    parser.add_argument("--bas_formtask", type=str)
    parser.add_argument("--bas_sprecherinnen", type=str)
    parser.add_argument("--common_voice", type=str)
    parser.add_argument("--css_ten", type=str)
    parser.add_argument("--lingualibre", type=str)
    parser.add_argument("--mailabs", type=str)
    parser.add_argument("--swc", type=str)
    parser.add_argument("--tatoeba", type=str)
    parser.add_argument("--tuda", type=str)
    parser.add_argument("--voxforge", type=str)
    parser.add_argument("--zamia_speech", type=str)
    args = parser.parse_args()

    corpora = []

    if args.bas_formtask:
        print("Loading bas-formtask ...")
        dataset = load_basformtask(args.bas_formtask)
        write_dataset(dataset, args.target_path)
        return

    if args.bas_sprecherinnen:
        print("Loading bas-sprecherinnen ...")
        dataset = load_bassprecherinnen(args.bas_sprecherinnen)
        write_dataset(dataset, args.target_path)
        return

    if args.common_voice is not None:
        print("Loading common-voice ...")
        corpus = audiomate.Corpus.load(args.common_voice, reader="common-voice")
        corpora.append(corpus)

    if args.css_ten is not None:
        print("Loading css-ten ...")
        corpus = audiomate.Corpus.load(args.css_ten, reader="css10")
        corpora.append(corpus)

    if args.lingualibre is not None:
        print("Loading lingualibre ...")
        corpus = audiomate.Corpus.load(args.lingualibre, reader="lingualibre")
        corpora.append(corpus)

    if args.mailabs is not None:
        print("Loading mailabs ...")
        corpus = audiomate.Corpus.load(args.mailabs, reader="mailabs")
        corpora.append(corpus)

    if args.tatoeba is not None:
        print("Loading tatoeba ...")
        corpus = audiomate.Corpus.load(args.tatoeba, reader="tatoeba")
        corpora.append(corpus)

    if args.swc is not None:
        print("Loading swc ...")
        corpus = audiomate.Corpus.load(args.swc, reader="swc")
        corpora.append(corpus)

    if args.tuda is not None:
        print("Loading tuda ...")
        corpus = audiomate.Corpus.load(args.tuda, reader="tuda")
        corpora.append(corpus)

    if args.voxforge is not None:
        print("Loading voxforge ...")
        corpus = audiomate.Corpus.load(args.voxforge, reader="voxforge")
        corpora.append(corpus)

    if args.zamia_speech is not None:
        print("Loading zamia-speech ...")
        corpus = audiomate.Corpus.load(args.zamia_speech, reader="zamia-speech")
        corpora.append(corpus)

    if len(corpora) <= 0:
        raise ValueError("No Corpus given!")

    merged_corpus = audiomate.Corpus.merge_corpora(corpora)
    clean_transcriptions(merged_corpus)

    print("Splitting corpus ...")
    splitter = subset.Splitter(merged_corpus, random_seed=38)
    split_sizes = {"train": 0.7, "dev": 0.15, "test": 0.15}

    splits = splitter.split(split_sizes, separate_issuers=True)
    if "test" not in splits or "dev" not in splits:
        print("Very small dataset -> using random split instead of separated speakers")
        splits = splitter.split(split_sizes, separate_issuers=False)

    merged_corpus.import_subview("train", splits["train"])
    merged_corpus.import_subview("dev", splits["dev"])
    merged_corpus.import_subview("test", splits["test"])

    print("Saving corpus ...")
    deepspeech_writer = io.MozillaDeepSpeechWriter()
    deepspeech_writer.save(merged_corpus, args.target_path)


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
