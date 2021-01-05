import os

import tensorflow as tf

from . import pipeline, training, utils

# ==================================================================================================

# Use growing gpu memory
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

config = utils.get_config()
checkpoint_dir = config["checkpoint_dir"]

model: tf.keras.Model

# ==================================================================================================


def run_test(dataset_test):
    print("\nEvaluating ...")
    loss = 0
    step = 0
    log_greedy_steps = config["log_prediction_steps"]

    for samples in dataset_test:
        features = samples["features"]
        # features = np.random.uniform(low=-1, high=1, size=[8, 211, 26]).astype(np.float32)
        predictions = model.predict(features)
        loss += training.get_loss(predictions, samples).numpy()
        step += 1

        if log_greedy_steps != 0 and step % log_greedy_steps == 0:
            training.log_greedy_text(predictions, samples)

    loss = loss / step
    print("Test loss: {}".format(loss))


# ==================================================================================================


def main():
    global model

    # Use exported config to set up the pipeline
    path = os.path.join(checkpoint_dir, "config_export.json")
    exported_config = utils.load_json_file(path)

    dataset_test = pipeline.create_pipeline(
        csv_path=config["data_paths"]["test"],
        batch_size=config["batch_sizes"]["test"],
        config=exported_config,
        augment=False,
        cache_path="",
    )

    # model = tf.saved_model.load(checkpoint_dir)
    model = tf.keras.models.load_model(checkpoint_dir)

    # Print network summary
    feature_type = exported_config["audio_features"]["use_type"]
    c_input = exported_config["audio_features"][feature_type]["num_features"]
    model.build(input_shape=(None, None, c_input))
    model.compile()
    model.summary()

    run_test(dataset_test)
