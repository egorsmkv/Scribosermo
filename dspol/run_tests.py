import tqdm

from dspol import pipeline, testing, utils

# ==================================================================================================


def test_pipeline():
    sample_csv_path = "/data_prepared/de/voxforge/test_azce.csv"
    augment = True
    bench_batch_size = 16
    config = utils.get_config()

    # Run pipeline for one epoch to check how long preprocessing takes
    print("\nGoing through dataset to check preprocessing duration...")
    tds = pipeline.create_pipeline(sample_csv_path, bench_batch_size, config, augment)
    for samples in tqdm.tqdm(tds):
        pass

    # Print the first two samples of the dataset
    tds = pipeline.create_pipeline(sample_csv_path, 2, config, augment)
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
    # test_pipeline()
