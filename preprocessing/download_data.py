import argparse
import os

from audiomate.corpus import io

# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training.")
    parser.add_argument("target_path", type=str)
    parser.add_argument("--tuda", action="store_true")
    parser.add_argument("--voxforge", action="store_true")
    parser.add_argument("--swc", action="store_true")
    parser.add_argument("--mailabs", action="store_true")
    parser.add_argument("--common_voice", action="store_true")
    parser.add_argument("--tatoeba", action="store_true")
    parser.add_argument("--zamia_speech", action="store_true")
    args = parser.parse_args()

    if args.tuda:
        print("Downloading tuda ...")
        dl = io.TudaDownloader()
        dl.download(os.path.join(args.target_path, "tuda"))

    if args.voxforge:
        print("Downloading voxforge ...")
        dl = io.VoxforgeDownloader(lang="de")
        dl.download(os.path.join(args.target_path, "voxforge"))

    if args.swc:
        print("Downloading swc ...")
        dl = io.SWCDownloader(lang="de")
        dl.download(os.path.join(args.target_path, "swc"))

    if args.mailabs:
        print("Downloading mailabs ...")
        dl = io.MailabsDownloader(tags=["de_DE"])
        dl.download(os.path.join(args.target_path, "mailabs"))

    if args.common_voice:
        print("Downloading common-voice ...")
        dl = io.CommonVoiceDownloader(lang="de")
        dl.download(os.path.join(args.target_path, "common_voice"))

    if args.tatoeba:
        print("Downloading tatoeba ...")
        dl = io.TatoebaDownloader(include_languages=["deu"])
        dl.download(os.path.join(args.target_path, "tatoeba"))

    if args.zamia_speech:
        print("Downloading zamia-speech ...")
        dl = io.ZamiaSpeechDownloader(lang="de")
        dl.download(os.path.join(args.target_path, "zamia_speech"))


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
