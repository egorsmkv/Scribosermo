import argparse

import downloaders

# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training.")
    parser.add_argument("target_path", type=str)
    parser.add_argument("--cv_singleword", action="store_true")
    parser.add_argument("--terrax", action="store_true")
    parser.add_argument("--ykollektiv", action="store_true")
    args = parser.parse_args()

    if args.cv_singleword:
        print("Downloading cv-singleword ...")
        link = "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com/cv-corpus-5-singleword/cv-corpus-5-singleword.tar.gz"
        downloaders.download_and_extract(link, args.target_path)

    if args.terrax:
        print("Downloading terra-x ...")
        link = (
            "https://www.youtube.com/watch?v=DL4faBZhHuo&list=UUA3mpqm67CpJ13YfA8qAnow"
        )
        downloaders.download_youtube_playlist(args.target_path, link, "de")

    if args.ykollektiv:
        print("Downloading y-kollektiv ...")
        link = (
            "https://www.youtube.com/watch?v=pbIhmqRhvFo&list=UULoWcRy-ZjA-Erh0p_VDLjQ"
        )
        downloaders.download_youtube_playlist(args.target_path, link, "de")


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
