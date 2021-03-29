import argparse
import os

from corcua import downloaders

# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Prepare data for training.")
    parser.add_argument("--target_path", type=str, required=True)
    parser.add_argument("--kurzgesagt", action="store_true")
    parser.add_argument("--musstewissen", action="store_true")
    parser.add_argument("--pulsreportage", action="store_true")
    parser.add_argument("--terrax", action="store_true")
    parser.add_argument("--ykollektiv", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.target_path):
        os.makedirs(args.target_path, exist_ok=True)

    if args.kurzgesagt:
        print("Downloading kurzgesagt ...")
        link = (
            "https://www.youtube.com/watch?v=erDUXM8mCS8&list=UUwRH985XgMYXQ6NxXDo8npw"
        )
        path = os.path.join(args.target_path, "kurzgesagt")
        downloaders.youtube.Downloader().download_dataset(
            path=path, overwrite=True, args={"link": link, "lang": "de"}
        )

    if args.musstewissen:
        print("Downloading musstewissen-deutsch ...")
        link = (
            "https://www.youtube.com/watch?v=ZnQiXDyHJcY&list=UUzOHLoNwbebvEkn7y6x-EWA"
        )
        path = os.path.join(args.target_path, "musstewissen_deutsch")
        downloaders.youtube.Downloader().download_dataset(
            path=path, overwrite=True, args={"link": link, "lang": "de"}
        )

        print("\nDownloading musstewissen-mathe ...")
        link = (
            "https://www.youtube.com/watch?v=ba-uUUaDxSo&list=UUaxX8488TqU6bZdcKpxPVvQ"
        )
        path = os.path.join(args.target_path, "musstewissen_mathe")
        downloaders.youtube.Downloader().download_dataset(
            path=path, overwrite=True, args={"link": link, "lang": "de"}
        )

        print("\nDownloading musstewissen-physik ...")
        link = (
            "https://www.youtube.com/watch?v=CSwivAbOhio&list=UU9RSWjfMU3qMixhigyHjEgw"
        )
        path = os.path.join(args.target_path, "musstewissen_physik")
        downloaders.youtube.Downloader().download_dataset(
            path=path, overwrite=True, args={"link": link, "lang": "de"}
        )

        print("\nDownloading musstewissen-chemie ...")
        link = (
            "https://www.youtube.com/watch?v=Mp9ss59KoWI&list=UU146qqkUMTrn4nfSSOTNwiA"
        )
        path = os.path.join(args.target_path, "musstewissen_chemie")
        downloaders.youtube.Downloader().download_dataset(
            path=path, overwrite=True, args={"link": link, "lang": "de"}
        )

    if args.pulsreportage:
        print("Downloading puls-reportage ...")
        link = (
            "https://www.youtube.com/watch?v=yl9B3KVqwQs&list=UUBzai1GXVKDdVCrwlKZg_6Q"
        )
        path = os.path.join(args.target_path, "pulsreportage")
        downloaders.youtube.Downloader().download_dataset(
            path=path, overwrite=True, args={"link": link, "lang": "de"}
        )

    if args.terrax:
        print("Downloading terra-x ...")
        link = (
            "https://www.youtube.com/watch?v=DL4faBZhHuo&list=UUA3mpqm67CpJ13YfA8qAnow"
        )
        path = os.path.join(args.target_path, "terrax")
        downloaders.youtube.Downloader().download_dataset(
            path=path, overwrite=True, args={"link": link, "lang": "de"}
        )

    if args.ykollektiv:
        print("Downloading y-kollektiv ...")
        link = (
            "https://www.youtube.com/watch?v=pbIhmqRhvFo&list=UULoWcRy-ZjA-Erh0p_VDLjQ"
        )
        path = os.path.join(args.target_path, "ykollektiv")
        downloaders.youtube.Downloader().download_dataset(
            path=path, overwrite=True, args={"link": link, "lang": "de"}
        )


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
