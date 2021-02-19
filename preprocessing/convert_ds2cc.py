import argparse
import os
import re

from corcua import readers, writers

# ==================================================================================================

pattern_text_commas = (
    r"((?<!(.wav))(?<!(wav_filename)|(wav_filesize))"
    r"(?<!(.wav,[0-9]{1}))(?<!(.wav,[0-9]{2}))(?<!(.wav,[0-9]{3}))(?<!(.wav,[0-9]{4}))"
    r"(?<!(.wav,[0-9]{5}))(?<!(.wav,[0-9]{6}))(?<!(.wav,[0-9]{7}))),"
)

# ==================================================================================================


def load_and_write(args, file, overwrite):
    spath = os.path.join(args.source_path, file)

    if args.remove_text_commas:
        print("Replacing commas in transcriptions ...")
        with open(spath, "r", encoding="utf-8") as cfile:
            content = cfile.read()
            content = re.sub(pattern_text_commas, " ", content)
        spath = "/tmp/ds2cc.csv"
        with open(spath, "w+", encoding="utf-8") as cfile:
            cfile.write(content)

    apath = os.path.join(args.target_path, "all.csv")
    tpath = os.path.join(args.target_path, file)
    ds = readers.deepspeech.Reader().load_dataset({"path_csv": spath})
    writers.base_writer.Writer().save_dataset(
        ds, args.target_path, sample_rate=16000, overwrite=overwrite
    )
    os.rename(apath, tpath)


# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert from deepspeech to corcua format"
    )
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--dev", type=str)
    parser.add_argument("--test", type=str)
    parser.add_argument("--remove_text_commas", action="store_true")
    args = parser.parse_args()

    if args.source_path == args.target_path:
        raise ValueError("Source and Target must be different")

    load_and_write(args, args.train, overwrite=True)
    if args.dev:
        load_and_write(args, args.dev, overwrite=False)
    if args.test:
        load_and_write(args, args.test, overwrite=False)


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
