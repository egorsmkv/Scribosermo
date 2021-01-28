import tqdm

from dspol import pipeline, testing, utils

# ==================================================================================================


def test_pipeline(sample_csv_path):
    train_mode = True
    bench_batch_size = 16
    config = utils.get_config()

    # Run pipeline for one epoch to check how long preprocessing takes
    print("\nGoing through dataset to check preprocessing duration...")
    tds = pipeline.create_pipeline(
        sample_csv_path, bench_batch_size, config, train_mode
    )
    for _ in tqdm.tqdm(tds):
        pass

    # Print the first sample of the dataset
    tds = pipeline.create_pipeline(sample_csv_path, 1, config, train_mode)
    for samples in tds:
        print(samples)
        break


# ==================================================================================================


def print_config():
    config = utils.get_config()
    print(config)


# ==================================================================================================

if __name__ == "__main__":
    print("\n======================================================================\n")

    testing.main()

    # print_config()
    # test_pipeline("/data_prepared/en/librispeech/test-clean_azce.csv")
    print("FINISHED")
