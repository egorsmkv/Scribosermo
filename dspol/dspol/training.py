import json
import os
import shutil
import time

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tqdm
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from . import nets, pipeline, utils

# ==================================================================================================

# Use growing gpu memory
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

config = utils.get_config()
checkpoint_dir = config["checkpoint_dir"]
cache_dir = config["cache_dir"]

alphabet = utils.load_alphabet(config)
idx2char = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant([i for i, u in enumerate(alphabet)]),
        values=tf.constant([u for i, u in enumerate(alphabet)]),
    ),
    default_value=tf.constant(" "),
)

model: tf.keras.Model
summary_writer: tf.summary.SummaryWriter
save_manager: tf.train.CheckpointManager
optimizer: tf.keras.optimizers.Adam


# ==================================================================================================


@tf.function(experimental_relax_shapes=True)
def loss_function(predictions, logit_lengths, samples):
    labels = samples["label"]
    label_lengths = samples["label_length"]

    # Blank index of "-1" returned better results compared to labels starting from 1, reason unclear
    loss = tf.nn.ctc_loss(
        labels=labels,
        logits=predictions,
        label_length=label_lengths,
        logit_length=logit_lengths,
        blank_index=-1,
        logits_time_major=False,
    )

    loss = tf.reduce_mean(loss)
    return loss


# ==================================================================================================


def get_loss(predictions, samples):
    # Calculate logit length here, that we can decorate the loss calculation
    # with a tf-function call for much faster calculations
    logit_lengths = tf.constant(
        tf.shape(predictions)[1], shape=tf.shape(predictions)[0]
    )
    # logit_lengths = samples["feature_length"]
    loss = loss_function(predictions, logit_lengths, samples)
    # loss = optimizer.get_scaled_loss(loss)
    return loss


# ==================================================================================================

# @tf.function
def train_step(samples, step):
    features = samples["features"]

    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = get_loss(predictions, samples)

    with summary_writer.as_default():
        tf.summary.experimental.set_step(step)
        tf.summary.scalar("loss", loss)

    trainable_variables = model.trainable_variables
    gradients = tape.gradient(loss, model.trainable_variables)
    # gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, predictions


# ==================================================================================================


def log_greedy_text(predictions, samples):

    # Drop all except first sample, and switch batch_size and time_steps
    predictions = tf.expand_dims(predictions[0], axis=0)
    predictions = tf.transpose(predictions, perm=[1, 0, 2])

    logit_lengths = tf.constant(tf.shape(predictions)[0], shape=(1,))
    decoded = tf.nn.ctc_greedy_decoder(predictions, logit_lengths, merge_repeated=True)

    label = samples["label"][0]
    label = idx2char.lookup(label).numpy()
    label = b"".join(label).strip()
    print("=Label======: {}".format(label))

    values = tf.cast(decoded[0][0].values, dtype=tf.int32)
    values = idx2char.lookup(values).numpy()
    values = b"".join(values)
    print("=Prediction=: {}".format(values))


# ==================================================================================================


def train(dataset_train, dataset_val, start_epoch, stop_epoch):
    step = np.int64(0)
    log_greedy_steps = config["log_prediction_steps"]
    # tf.profiler.experimental.start('/checkpoints/profiles/')

    for epoch in range(start_epoch, stop_epoch):
        start_time = time.time()

        for samples in dataset_train:

            # tf.summary.trace_on(graph=True, profiler=False)

            # tf.profiler.experimental.client.trace('grpc://localhost:6009',
            #                                       checkpoint_dir, 2000)

            # with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
            #     # Not recommended, but couldn't get next dataset element here
            #     loss, predictions = train_step(samples, step)

            loss, predictions = train_step(samples, step)
            step += 1
            print("Step: {} - Epoch: {} - Loss: {}".format(step, epoch, loss.numpy()))

            # with summary_writer.as_default():
            #     tf.summary.trace_export(
            #         name="my_func_trace",
            #         step=0)

            if log_greedy_steps != 0 and step % log_greedy_steps == 0:
                # print("")
                # print(samples["features"][0])
                # print(samples["label"][0])
                # print(predictions)
                log_greedy_text(predictions, samples)

        save_manager.save()
        # eval(dataset_val)
        # tf.saved_model.save(model, checkpoint_dir)
        tf.keras.models.save_model(model, checkpoint_dir, include_optimizer=False)

        msg = "Epoch {} took {} hours\n"
        duration = utils.seconds_to_hours(time.time() - start_time)
        print(msg.format(epoch, duration))

    # tf.profiler.experimental.stop()


# ==================================================================================================


def eval(dataset_val):
    print("\nEvaluating ...")
    loss = 0
    step = 0
    log_greedy_steps = config["log_prediction_steps"]

    for samples in dataset_val:
        features = samples["features"]
        predictions = model(features)
        loss += get_loss(predictions, samples).numpy()
        step += 1

        if log_greedy_steps != 0 and step % log_greedy_steps == 0:
            log_greedy_text(predictions, samples)

    loss = loss / step
    print("Validation loss: {}".format(loss))


# ==================================================================================================


def main():
    global model, summary_writer, save_manager, optimizer

    # Delete old data and create folders
    if os.path.exists(checkpoint_dir):
        utils.delete_dir(checkpoint_dir)
    if os.path.exists(cache_dir):
        utils.delete_dir(cache_dir)
    os.makedirs(checkpoint_dir)
    os.makedirs(cache_dir)

    # Export current config next to the checkpoints
    path = os.path.join(checkpoint_dir, "config_export.json")
    with open(path, "w+", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    # Create pipelines
    cache = config["cache_dir"] + "train" if config["use_pipeline_cache"] else ""
    dataset_train = pipeline.create_pipeline(
        csv_path=config["data_paths"]["train"],
        batch_size=config["batch_sizes"]["train"],
        config=config,
        augment=True,
        cache_path=cache,
    )
    cache = config["cache_dir"] + "val" if config["use_pipeline_cache"] else ""
    dataset_val = pipeline.create_pipeline(
        csv_path=config["data_paths"]["val"],
        batch_size=config["batch_sizes"]["val"],
        config=config,
        augment=False,
        cache_path=cache,
    )

    # tf.profiler.experimental.server.start(6009)
    # tf.summary.trace_on(graph=True, profiler=True)
    # tf.summary.trace_on(graph=True, profiler=False)

    feature_type = config["audio_features"]["use_type"]
    c_input = config["audio_features"][feature_type]["num_features"]
    c_output = len(alphabet) + 1

    # Build the network
    network_type = config["network"]["name"]
    if network_type == "deepspeech1":
        model = nets.deepspeech1.MyModel(c_input, c_output)
    elif network_type == "deepspeech2":
        model = nets.deepspeech2.MyModel(c_input, c_output)
    elif network_type == "jasper":
        model = nets.jasper.MyModel(
            c_input,
            c_output,
            blocks=config["network"]["blocks"],
            module_repeat=config["network"]["module_repeat"],
            dense_residuals=config["network"]["dense_residuals"],
        )
    elif network_type == "quartznet":
        model = nets.quartznet.MyModel(
            c_input,
            c_output,
            blocks=config["network"]["blocks"],
            module_repeat=config["network"]["module_repeat"],
        )

    # Print network summary
    model.build(input_shape=(None, None, c_input))
    model.summary()
    # tf.keras.models.save_model(model, checkpoint_dir, include_optimizer=False)

    # Select optimizer
    optimizer_type = config["optimizer"]["name"]
    if optimizer_type == "adamw":
        optimizer = tfa.optimizers.AdamW(
            learning_rate=config["optimizer"]["learning_rate"],
            weight_decay=config["optimizer"]["weight_decay"],
        )
    elif optimizer_type == "novograd":
        optimizer = tfa.optimizers.NovoGrad(
            learning_rate=config["optimizer"]["learning_rate"],
            weight_decay=config["optimizer"]["weight_decay"],
            beta_1=config["optimizer"]["beta_1"],
            beta_2=config["optimizer"]["beta_2"],
        )
    # optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

    summary_writer = tf.summary.create_file_writer(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    save_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=1
    )

    # Load old checkpoint and its epoch number if existing
    start_epoch = 0
    if save_manager.latest_checkpoint:
        start_epoch = int(save_manager.latest_checkpoint.split("-")[-1])
        checkpoint.restore(save_manager.latest_checkpoint)
    start_epoch += 1

    # Finally the training can start
    max_epoch = config["training_epochs"]
    train(dataset_train, dataset_val, start_epoch, max_epoch)
