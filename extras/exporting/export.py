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
    "use_fixed_norm": False,
}

checkpoint_dir = "/checkpoints/en/qnetp15/"
export_dir = os.path.join(checkpoint_dir, "exported/")

# ==================================================================================================


def main():
    nn_model = tf.keras.models.load_model(checkpoint_dir)
    model = exmodel.MyModel(nn_model, metadata)

    model.build(input_shape=(None, None))
    # model.build(input_shape=(None, 73152))
    model.compile()
    model.summary()

    # Export as .pb model
    tf.keras.models.save_model(model, export_dir + "pb/", include_optimizer=False)

    # Export as .tflite model, extend supported ops to be able to create the spectrogram
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_model = converter.convert()
    with open(export_dir + "model.tflite", "wb+") as file:
        file.write(tflite_model)


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
