import multiprocessing as mp
import os
import shutil
import sys
from functools import partial

import pandas as pd
import tqdm
from pydub import AudioSegment

# The if block is required for isort
if True:  # pylint: disable=using-constant-test
    file_path = os.path.dirname(os.path.realpath(__file__)) + "/"
    sys.path.insert(1, file_path + "../preprocessing")
    import text_cleaning


# ==================================================================================================


def write_dataset(dataset, path):
    """Saves a dataset in DeepSpeech format.
    Input a list of dicts containing at least 'file' and 'transcription' keys."""
    print("\nSaving dataset ...")

    # Create or delete and recreate target directory
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    path_audios = os.path.join(path, "audios")
    os.makedirs(path_audios, exist_ok=True)

    cc_func = partial(convert_and_clean, target_path=path_audios)
    with mp.Pool(mp.cpu_count()) as p:
        new_dataset = list(tqdm.tqdm(p.imap(cc_func, dataset), total=len(dataset)))
    new_dataset = [d for d in new_dataset if d is not None]

    dataset = pd.DataFrame(new_dataset)
    dataset.to_csv(os.path.join(path, "all.csv"), index=False, encoding="utf-8")


# ==================================================================================================


def convert_and_clean(entry, target_path):
    if not os.path.exists(entry["file"]):
        print("This file does not exist: {}".format(entry["file"]))
        return None

    name, extension = os.path.splitext(os.path.basename(entry["file"]))
    out_file = os.path.join(target_path, name + ".wav")

    audio = AudioSegment.from_file(entry["file"], extension)
    audio = audio.set_frame_rate(16000)
    audio = audio.set_channels(1)
    audio.export(out_file, format="wav")

    cleaned_transcript = text_cleaning.clean_sentence(entry["transcription"])[0]

    new_entry = {
        "wav_filename": out_file,
        "wav_filesize": os.path.getsize(out_file),
        "transcript": cleaned_transcript,
    }
    return new_entry
