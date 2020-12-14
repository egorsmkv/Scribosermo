import tqdm

from dspol import pipeline

# ==================================================================================================


def main():
    sample_csv_path = "/data_prepared/de/voxforge/test_azce.csv"
    feature_type = "mfcc"
    feature_type = "lfbank"
    augment = True
    pipeline.delete_cache()

    # Run pipeline for one epoch to check how long preprocessing takes
    print("\nGoing through dataset to check preprocessing duration...")
    tds, _ = pipeline.create_pipeline(sample_csv_path, 16, feature_type, augment)
    for samples in tqdm.tqdm(tds):
        pass
    pipeline.delete_cache()

    tds, _ = pipeline.create_pipeline(sample_csv_path, 2, feature_type, augment)
    for samples in tds:
        print(samples)
        print(samples["features"].shape[1])
        break


# ==================================================================================================

if __name__ == "__main__":
    main()
