import argparse
import os

import tensorflow as tf
from tensorflow.python.framework.ops import (
    disable_eager_execution,
    enable_eager_execution,
)

import model as exmodel
from scode import training, utils

# ==================================================================================================

metadata = {
    "network": "Quartznet",
    "num_features": 64,
    "num_mfcc_features": 0,
    "num_spectrogram_bins": 257,
    "audio_sample_rate": 16000,
    "audio_window_samples": int(16000 * 0.02),
    "audio_step_samples": int(16000 * 0.01),
    "use_volume_norm": False,
}

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
    parser = argparse.ArgumentParser(description="Export model")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--export_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    args = parser.parse_args()

    if args.mode == "pb":
        # Disabling eager execution solves a problem with AudioSpectrogram layer
        disable_eager_execution()
    elif args.mode == "tflite":
        # Eager execution is required for tflite
        enable_eager_execution()
    else:
        raise ValueError()

    # Load exported model
    nn_model = training.load_exported_model(args.checkpoint_dir)

    if args.mode == "pb":
        # Export as .pb model
        model_pb = exmodel.MyModel(nn_model, metadata, specmode="pb")
        model_pb.build(input_shape=(None, None))
        model_pb.summary()

        tf.keras.models.save_model(
            model_pb, args.export_dir + "pb/", include_optimizer=False
        )

    elif args.mode == "tflite":
        # Export as .tflite model
        model_tl = exmodel.MyModel(nn_model, metadata, specmode="tflite")
        model_tl.build(input_shape=(None, None))
        model_tl.summary()

        export_tflite(model_tl, args.export_dir + "model_full.tflite", optimize=False)
        export_tflite(
            model_tl, args.export_dir + "model_quantized.tflite", optimize=True
        )


# ==================================================================================================

if __name__ == "__main__":
    main()
    print("FINISHED")
