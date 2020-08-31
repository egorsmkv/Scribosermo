import argparse
import os

import downloaders

# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training.")
    parser.add_argument("target_path", type=str)
    parser.add_argument("--cv_singleword", action="store_true")
    parser.add_argument("--kurzgesagt", action="store_true")
    parser.add_argument("--musstewissen", action="store_true")
    parser.add_argument("--pulsreportage", action="store_true")
    parser.add_argument("--terrax", action="store_true")
    parser.add_argument("--ykollektiv", action="store_true")
    args = parser.parse_args()

    if args.cv_singleword:
        print("Downloading cv-singleword ...")
        link = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-singleword/cv-corpus-5-singleword.tar.gz"
        downloaders.download_and_extract(link, args.target_path)

    if args.kurzgesagt:
        print("Downloading kurzgesagt ...")
        link = (
            "https://www.youtube.com/watch?v=erDUXM8mCS8&list=UUwRH985XgMYXQ6NxXDo8npw"
        )
        path = os.path.join(args.target_path, "kurzgesagt")
        downloaders.download_youtube_playlist(path, link, "de")

    if args.musstewissen:
        print("Downloading musstewissen-deutsch ...")
        link = (
            "https://www.youtube.com/watch?v=ZnQiXDyHJcY&list=UUzOHLoNwbebvEkn7y6x-EWA"
        )
        path = os.path.join(args.target_path, "musstewissen_deutsch")
        downloaders.download_youtube_playlist(path, link, "de")

        print("\nDownloading musstewissen-mathe ...")
        link = (
            "https://www.youtube.com/watch?v=ba-uUUaDxSo&list=UUaxX8488TqU6bZdcKpxPVvQ"
        )
        path = os.path.join(args.target_path, "musstewissen_mathe")
        downloaders.download_youtube_playlist(path, link, "de")

        print("\nDownloading musstewissen-physik ...")
        link = (
            "https://www.youtube.com/watch?v=CSwivAbOhio&list=UU9RSWjfMU3qMixhigyHjEgw"
        )
        path = os.path.join(args.target_path, "musstewissen_physik")
        downloaders.download_youtube_playlist(path, link, "de")

        print("\nDownloading musstewissen-chemie ...")
        link = (
            "https://www.youtube.com/watch?v=Mp9ss59KoWI&list=UU146qqkUMTrn4nfSSOTNwiA"
        )
        path = os.path.join(args.target_path, "musstewissen_chemie")
        downloaders.download_youtube_playlist(path, link, "de")

    if args.pulsreportage:
        print("Downloading puls-reportage ...")
        link = (
            "https://www.youtube.com/watch?v=yl9B3KVqwQs&list=UUBzai1GXVKDdVCrwlKZg_6Q"
        )
        path = os.path.join(args.target_path, "pulsreportage")
        downloaders.download_youtube_playlist(path, link, "de")

    if args.terrax:
        print("Downloading terra-x ...")
        link = (
            "https://www.youtube.com/watch?v=DL4faBZhHuo&list=UUA3mpqm67CpJ13YfA8qAnow"
        )
        path = os.path.join(args.target_path, "terrax")
        downloaders.download_youtube_playlist(path, link, "de")

    if args.ykollektiv:
        print("Downloading y-kollektiv ...")
        link = (
            "https://www.youtube.com/watch?v=pbIhmqRhvFo&list=UULoWcRy-ZjA-Erh0p_VDLjQ"
        )
        path = os.path.join(args.target_path, "ykollektiv")
        downloaders.download_youtube_playlist(path, link, "de")


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
