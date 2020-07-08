import argparse
import os

from audiomate.corpus import io


# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training.")
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--common_voice", action="store_true")
    parser.add_argument("--css_ten", action="store_true")
    parser.add_argument("--lingualibre", action="store_true")
    parser.add_argument("--mailabs", action="store_true")
    parser.add_argument("--swc", action="store_true")
    parser.add_argument("--tatoeba", action="store_true")
    parser.add_argument("--tuda", action="store_true")
    parser.add_argument("--voxforge", action="store_true")
    parser.add_argument("--zamia_speech", action="store_true")
    args = parser.parse_args()

    if args.common_voice:
        dl = io.CommonVoiceDownloader(lang=args.language)
        print("Downloading common-voice-{} ...".format(args.language))
        dl.download(os.path.join(args.target_path, "common_voice"))

    if args.lingualibre:
        langs = {
            "de": "deu",
            "eo": "epo",
            "es": "spa",
            "fr": "fra",
            "it": "ita",
            "pl": "pol",
        }
        if args.language in langs:
            print("Downloading lingualibre-{} ...".format(args.language))
            dl = io.LinguaLibreDownloader(lang=langs[args.language])
            dl.download(os.path.join(args.target_path, "lingualibre"))
        else:
            msg = "Language '{}' not supported for lingualibre"
            raise ValueError(msg.format(args.language))

    if args.mailabs:
        langs = {
            "de": "de_DE",
            "es": "es_ES",
            "fr": "fr_FR",
            "it": "it_IT",
            "pl": "pl_PL",
            "ru": "ru_RU",
            "uk": "uk_UK",
        }
        if args.language in langs:
            print("Downloading mailabs-{} ...".format(args.language))
            dl = io.MailabsDownloader(lang=langs[args.language])
            dl.download(os.path.join(args.target_path, "mailabs"))
        else:
            msg = "Language '{}' not supported for mailabs"
            raise ValueError(msg.format(args.language))

    if args.swc:
        dl = io.SWCDownloader(lang=args.language)
        print("Downloading swc-{} ...".format(args.language))
        dl.download(os.path.join(args.target_path, "swc"))

    if args.tatoeba:
        langs = {
            "de": "deu",
            "eo": "epo",
            "es": "spa",
            "fr": "fra",
            "it": "ita",
            "pl": "pol",
        }
        if args.language in langs:
            print("Downloading tatoeba-{} ...".format(args.language))
            dl = io.TatoebaDownloader(include_languages=[langs[args.language]])
            dl.download(os.path.join(args.target_path, "tatoeba"))
        else:
            msg = "Language '{}' not supported for tatoeba"
            raise ValueError(msg.format(args.language))

    if args.tuda:
        print("Downloading tuda-de ...".format(args.language))
        dl = io.TudaDownloader()
        dl.download(os.path.join(args.target_path, "tuda"))

    if args.voxforge:
        dl = io.VoxforgeDownloader(lang=args.language)
        print("Downloading voxforge-{} ...".format(args.language))
        dl.download(os.path.join(args.target_path, "voxforge"))

    if args.zamia_speech:
        dl = io.ZamiaSpeechDownloader(lang=args.language)
        print("Downloading zamia-speech-{} ...".format(args.language))
        dl.download(os.path.join(args.target_path, "zamia_speech"))


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
