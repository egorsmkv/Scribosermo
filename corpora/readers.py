import json
import os
import re

import tqdm

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


def load_youtube(path):
    print("\nLoading transcripts ...")

    align_path = os.path.join(path, "alignment.json")
    with open(align_path, "r", encoding="utf-8") as file:
        aligns = json.load(file)

    dataset = []
    for a in tqdm.tqdm(aligns):
        entry = {
            "file": a,
            "transcription": aligns[a],
        }
        dataset.append(entry)

    return dataset
