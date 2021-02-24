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
    "use_volume_norm": False,
}

checkpoint_dir = "/checkpoints/en/qnetp15/"
export_dir = os.path.join(checkpoint_dir, "exported/")

# ==================================================================================================


def export_tflite(model, save_path, optimize):

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if optimize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(save_path, "wb+") as file:
        file.write(tflite_model)


# ==================================================================================================


def main():
    nn_model = tf.keras.models.load_model(checkpoint_dir)
    model_pb = exmodel.MyModel(nn_model, metadata, specmode="pb")
    model_tl = exmodel.MyModel(nn_model, metadata, specmode="tflite")

    model_pb.build(input_shape=(None, None))
    model_tl.build(input_shape=(None, None))
    model_pb.summary()

    # Export as .pb model
    tf.keras.models.save_model(model_pb, export_dir + "pb/", include_optimizer=False)

    # Export as .tflite model
    export_tflite(model_tl, export_dir + "model_full.tflite", optimize=False)
    export_tflite(model_tl, export_dir + "model_quantized.tflite", optimize=True)


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
