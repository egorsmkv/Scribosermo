#! /usr/bin/env python

import argparse
import os

from audiomate.corpus import io

# ======================================================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare data for training.')
    parser.add_argument('target_path', type=str)
    parser.add_argument('--tuda', action='store_true')
    parser.add_argument('--voxforge', action='store_true')
    parser.add_argument('--swc', action='store_true')
    parser.add_argument('--mailabs', action='store_true')
    parser.add_argument('--common_voice', action='store_true')

    args = parser.parse_args()

    tuda_path = args.tuda
    voxforge_path = args.voxforge
    swc_path = args.swc
    mailabs_path = args.mailabs
    cv_path = args.common_voice

    if tuda_path:
        print("Downloading tuda ...")
        dl = io.TudaDownloader()
        dl.download(os.path.join(args.target_path, "tuda"))

    if voxforge_path:
        print("Downloading voxforge ...")
        dl = io.VoxforgeDownloader(lang='de')
        dl.download(os.path.join(args.target_path, "voxforge"))

    if swc_path:
        print("Downloading swc ...")
        dl = io.SWCDownloader(lang='de')
        dl.download(os.path.join(args.target_path, "swc"))

    if mailabs_path:
        print("Downloading mailabs ...")
        dl = io.MailabsDownloader(tags=['de_DE'])
        dl.download(os.path.join(args.target_path, "mailabs"))

    if cv_path:
        print("No downloader for common voice -> Download manually")
