import argparse
import os

from audiomate.corpus import io

# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training.")
    parser.add_argument("target_path", type=str)
    parser.add_argument("--common_voice_de", action="store_true")
    parser.add_argument("--common_voice_fr", action="store_true")
    parser.add_argument("--mailabs_de", action="store_true")
    parser.add_argument("--mailabs_fr", action="store_true")
    parser.add_argument("--swc_de", action="store_true")
    parser.add_argument("--tatoeba_de", action="store_true")
    parser.add_argument("--tatoeba_fr", action="store_true")
    parser.add_argument("--tuda_de", action="store_true")
    parser.add_argument("--voxforge_de", action="store_true")
    parser.add_argument("--voxforge_fr", action="store_true")
    parser.add_argument("--zamia_speech_de", action="store_true")
    args = parser.parse_args()

    if args.common_voice_de:
        print("Downloading common-voice-de ...")
        dl = io.CommonVoiceDownloader(lang="de")
        dl.download(os.path.join(args.target_path, "common_voice"))

    if args.common_voice_fr:
        print("Downloading common-voice-fr ...")
        dl = io.CommonVoiceDownloader(lang="fr")
        dl.download(os.path.join(args.target_path, "common_voice"))

    if args.mailabs_de:
        print("Downloading mailabs-de ...")
        dl = io.MailabsDownloader(tags=["de_DE"])
        dl.download(os.path.join(args.target_path, "mailabs"))

    if args.mailabs_fr:
        print("Downloading mailabs-fr ...")
        dl = io.MailabsDownloader(tags=["fr_FR"])
        dl.download(os.path.join(args.target_path, "mailabs"))

    if args.swc_de:
        print("Downloading swc-de ...")
        dl = io.SWCDownloader(lang="de")
        dl.download(os.path.join(args.target_path, "swc"))

    if args.tatoeba_de:
        print("Downloading tatoeba-de ...")
        dl = io.TatoebaDownloader(include_languages=["deu"])
        dl.download(os.path.join(args.target_path, "tatoeba"))

    if args.tatoeba_fr:
        print("Downloading tatoeba-fr ...")
        dl = io.TatoebaDownloader(include_languages=["fra"])
        dl.download(os.path.join(args.target_path, "tatoeba"))

    if args.tuda_de:
        print("Downloading tuda-de ...")
        dl = io.TudaDownloader()
        dl.download(os.path.join(args.target_path, "tuda"))

    if args.voxforge_de:
        print("Downloading voxforge-de ...")
        dl = io.VoxforgeDownloader(lang="de")
        dl.download(os.path.join(args.target_path, "voxforge"))

    if args.voxforge_fr:
        print("Downloading voxforge-fr ...")
        dl = io.VoxforgeDownloader(lang="fr")
        dl.download(os.path.join(args.target_path, "voxforge"))

    if args.zamia_speech_de:
        print("Downloading zamia-speech-de ...")
        dl = io.ZamiaSpeechDownloader(lang="de")
        dl.download(os.path.join(args.target_path, "zamia_speech"))


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
