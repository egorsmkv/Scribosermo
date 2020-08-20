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
    dropped = 0
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

            # Drop files including signs for incomplete words or repetitions
            if any([s in trans for s in "*~"]):
                dropped += 1
                continue

            entry = {
                "file": wav,
                "transcription": trans,
                "speaker": os.path.basename(d),
            }
            dataset.append(entry)

    msg = "Dropped {}/{} files with speech errors"
    print(msg.format(dropped, len(dataset) + dropped))
    return dataset


# ==================================================================================================


def load_bassprecherinnen(path):
    print("Loading transcripts ...")
    dir_names = os.listdir(path)
    dir_names = [os.path.join(path, d) for d in dir_names]
    dir_names = [d for d in dir_names if os.path.isdir(d)]

    tags_pattern = re.compile("<.*?>")

    dataset = []
    dropped = 0
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

            # Drop files including signs for incomplete words or repetitions
            if any([s in trans for s in "*~"]):
                dropped += 1
                continue

            entry = {
                "file": wav,
                "transcription": trans,
                "speaker": os.path.basename(d),
            }
            dataset.append(entry)

    msg = "Dropped {}/{} files with speech errors"
    print(msg.format(dropped, len(dataset) + dropped))
    return dataset


# ==================================================================================================


def load_youtube(path):
    print("\nLoading transcripts ...")

    align_path = os.path.join(path, "alignment.json")
    with open(align_path, "r", encoding="utf-8") as file:
        aligns = json.load(file)

    dataset = []
    dropped = 0
    for a in tqdm.tqdm(aligns):
        # Drop files including signs for notes
        if any([aligns[a].startswith(s) for s in "*(["]):
            dropped += 1
            continue

        entry = {
            "file": a,
            "transcription": aligns[a],
        }
        dataset.append(entry)

    msg = "Dropped {}/{} files with notes"
    print(msg.format(dropped, len(aligns)))

    return dataset
