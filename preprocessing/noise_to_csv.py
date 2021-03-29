import os

# ==================================================================================================

noise_path = "/data_prepared/noise/"
noise_csv_path = noise_path + "all.csv"


# ==================================================================================================


def collect_file_texts(noise_dir):
    texts = []
    for dirpath, _, filenames in os.walk(noise_dir):
        for f in filenames:
            if f.endswith(".wav"):
                sound_path = os.path.join(dirpath, f)
                file_size = os.path.getsize(sound_path)
                text = "{},{},some noise".format(sound_path, file_size)
                texts.append(text)
    return texts


# ==================================================================================================


def main():
    files = collect_file_texts(noise_path)
    print("Found {} noise files".format(len(files)))

    text = "wav_filename,wav_filesize,transcript\n"
    text = text + "\n".join(files)
    with open(noise_csv_path, "w+", encoding="utf-8") as file:
        file.write(text + "\n")


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
