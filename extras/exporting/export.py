import os

import tensorflow as tf

import model as exmodel

# ==================================================================================================

metadata = {
    "network": "Quartznet",
    "num_features": 64,
    "num_spectrogram_bins": 257,
    "audio_sample_rate": 16000,
    "audio_window_samples": int(16000 * 0.02),
    "audio_step_samples": int(16000 * 0.01),
}

checkpoint_dir = "/checkpoints/en/tmp5/"

# ==================================================================================================


def main():
    nn_model = tf.keras.models.load_model(checkpoint_dir)
    model = exmodel.MyModel(nn_model, metadata)

    model.build(input_shape=(None, None))
    model.compile()
    model.summary()

    save_path = os.path.join(checkpoint_dir, "exported/pb")
    tf.keras.models.save_model(model, save_path, include_optimizer=False)


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
