"""
Taken from: https://github.com/mozilla/DeepSpeech/pull/2622#issuecomment-586853243
With some additions

Run with python 2
"""

from __future__ import absolute_import, division, print_function

import argparse
import math
import os
from functools import partial
from multiprocessing import Pool

import librosa
import tqdm
from pydub import AudioSegment

from dataset_operations import seconds_to_hours

# ==================================================================================================


def get_duration(filename):
    """Get duration of the wav file"""
    length = librosa.get_duration(filename=filename)
    return length


# ==================================================================================================


def detect_silence(
    sound: AudioSegment, silence_threshold=-50.0, chunk_size=10
) -> (int, int):
    start_trim = 0  # ms
    sound_size = len(sound)
    assert chunk_size > 0  # to avoid infinite loop
    while (
        sound[start_trim : (start_trim + chunk_size)].dBFS < silence_threshold
        and start_trim < sound_size
    ):
        start_trim += chunk_size

    end_trim = sound_size
    while (
        sound[(end_trim - chunk_size) : end_trim].dBFS < silence_threshold
        and end_trim > 0
    ):
        end_trim -= chunk_size

    start_trim = min(sound_size, start_trim)
    end_trim = max(0, end_trim)

    return min([start_trim, end_trim]), max([start_trim, end_trim])


# ==================================================================================================


def trim_silence_audio(
    sound: AudioSegment, silence_threshold=-50.0, chunk_size=10
) -> AudioSegment:
    start_trim, end_trim = detect_silence(sound, silence_threshold, chunk_size)
    return sound[start_trim:end_trim]


# ==================================================================================================


def convert(
    filename: str,
    dst_dirpath: str,
    dirpath: str,
    normalize: bool,
    trim_silence: bool,
    min_duration_seconds: float,
    max_duration_seconds: float,
):
    if not filename.endswith((".wav", ".raw")):
        return

    filepath = os.path.join(dirpath, filename)
    if filename.endswith(".wav"):
        sound: AudioSegment = AudioSegment.from_file(filepath)
    else:
        try:
            sound: AudioSegment = AudioSegment.from_raw(
                filepath, sample_width=2, frame_rate=44100, channels=1
            )
        except Exception as err:
            print("Retrying conversion: {}".format(err))
            try:
                sound: AudioSegment = AudioSegment.from_raw(
                    filepath, sample_width=2, frame_rate=48000, channels=1
                )
            except Exception as err:
                print("Skipping file {}, got error: {}".format(filepath, err))
                return
        try:
            sound = sound.set_frame_rate(16000)
        except Exception as err:
            print("Skipping {}".format(err))
            return

    n_splits: int = max(1, math.ceil(sound.duration_seconds / max_duration_seconds))
    chunk_duration_ms = math.ceil(len(sound) / n_splits)
    chunks = []

    for i in range(n_splits):
        end_ms = min((i + 1) * chunk_duration_ms, len(sound))
        chunk = sound[(i * chunk_duration_ms) : end_ms]
        chunks.append(chunk)

    for i, chunk in enumerate(chunks):
        dst_path = os.path.join(dst_dirpath, str(i) + "_" + filename)
        if dst_path.endswith(".raw"):
            dst_path = dst_path[:-4] + ".wav"

        if os.path.exists(dst_path):
            print("Audio already exists: {}".format(dst_path))
            return

        if normalize:
            chunk = chunk.normalize()
            if chunk.dBFS < -30.0:
                chunk = chunk.compress_dynamic_range().normalize()
            if chunk.dBFS < -30.0:
                chunk = chunk.compress_dynamic_range().normalize()
        if trim_silence:
            chunk = trim_silence_audio(chunk)

        if chunk.duration_seconds < min_duration_seconds:
            return
        chunk.export(dst_path, format="wav")


# ==================================================================================================


def get_noise_duration(dst_dir):
    dur = 0
    num = 0
    for dirpath, _, filenames in os.walk(dst_dir):
        for f in filenames:
            dur += get_duration(os.path.join(dirpath, f))
            num += 1
    return dur, num


# ==================================================================================================


def main(
    src_dir: str,
    dst_dir: str,
    min_duration_seconds: float,
    max_duration_seconds: float,
    normalize=True,
    trim_silence=True,
):
    assert os.path.exists(src_dir)
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=False)
    src_dir = os.path.abspath(src_dir)
    dst_dir = os.path.abspath(dst_dir)

    for dirpath, _, filenames in os.walk(src_dir):
        dirpath = os.path.abspath(dirpath)
        dst_dirpath = os.path.join(dst_dir, dirpath.replace(src_dir, "").lstrip("/"))

        print("Converting directory: {} -> {}".format(dirpath, dst_dirpath))
        if not os.path.exists(dst_dirpath):
            os.makedirs(dst_dirpath, exist_ok=False)

        convert_func = partial(
            convert,
            dst_dirpath=dst_dirpath,
            dirpath=dirpath,
            normalize=normalize,
            trim_silence=trim_silence,
            min_duration_seconds=min_duration_seconds,
            max_duration_seconds=max_duration_seconds,
        )

        with Pool(processes=None) as pool:
            for _ in tqdm.tqdm(
                pool.imap_unordered(convert_func, filenames), total=len(filenames)
            ):
                pass


# ==================================================================================================

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Optimize noise files")
    PARSER.add_argument("--from_dir", help="Convert wav from directory", type=str)
    PARSER.add_argument("--to_dir", help="save wav to directory", type=str)
    PARSER.add_argument(
        "--min_sec", help="min duration seconds of saved file", type=float, default=1.0
    )
    PARSER.add_argument(
        "--max_sec", help="max duration seconds of saved file", type=float, default=30.0
    )
    PARSER.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize sound range, default is true",
        default=True,
    )
    PARSER.add_argument(
        "--trim",
        action="store_true",
        help="Trim silence, default is true",
        default=True,
    )
    PARAMS = PARSER.parse_args()

    main(
        PARAMS.from_dir,
        PARAMS.to_dir,
        PARAMS.min_sec,
        PARAMS.max_sec,
        PARAMS.normalize,
        PARAMS.trim,
    )

    duration, file_num = get_noise_duration(PARAMS.to_dir)
    msg = "Your noise dataset has {} files and a duration of {} hours\n"
    print(msg.format(file_num, seconds_to_hours(duration)))
    print("FINISHED")
