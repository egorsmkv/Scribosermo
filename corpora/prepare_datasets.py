import argparse

import readers
import writer

# ==================================================================================================


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for training.")
    parser.add_argument("target_path", type=str)
    parser.add_argument("--bas_formtask", type=str)
    parser.add_argument("--bas_sprecherinnen", type=str)
    parser.add_argument("--youtube_dir", type=str)
    args = parser.parse_args()

    if args.bas_formtask:
        print("Loading bas-formtask ...")
        dataset = readers.load_basformtask(args.bas_formtask)
        writer.write_dataset(dataset, args.target_path)
        return

    if args.bas_sprecherinnen:
        print("Loading bas-sprecherinnen ...")
        dataset = readers.load_bassprecherinnen(args.bas_sprecherinnen)
        writer.write_dataset(dataset, args.target_path)
        return

    if args.youtube_dir:
        print("Loading youtube-directory ...")
        dataset = readers.load_youtube(args.youtube_dir)
        writer.write_dataset(dataset, args.target_path)
        return


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
