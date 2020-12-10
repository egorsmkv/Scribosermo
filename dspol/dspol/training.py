import os
import shutil
import time

import numpy as np
import tensorflow as tf
import tqdm
from tensorflow.keras.mixed_precision import experimental as mixed_precision

from . import nets, pipeline, utils

# ==================================================================================================

# Use growing gpu memory
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)


model: tf.keras.Model
summary_writer: tf.summary.SummaryWriter
save_manager: tf.train.CheckpointManager
optimizer: tf.keras.optimizers.Adam

batch_size = 1
max_epoch = 30
learning_rate = 0.0001
# dataset_train_path = "/data_prepared/de/voxforge/train_azce.csv"
# dataset_train_path = "/data_prepared/de/voxforge/mini_test_azce.csv"
dataset_train_path = "/data_prepared/de/voxforge/nano_test_azce.csv"
# dataset_val_path = "/data_prepared/de/voxforge/dev_azce.csv"
dataset_val_path = "/data_prepared/de/voxforge/nano_val_azce.csv"
dataset_test_path = "/data_prepared/de/voxforge/test_azce.csv"
checkpoint_dir = "/checkpoints/tmp/"

alphabet = " abcdefghijklmnopqrstuvwxyz'"
idx2char = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant([i for i, u in enumerate(alphabet)]),
        values=tf.constant([u for i, u in enumerate(alphabet)]),
    ),
    default_value=tf.constant(" "),
)


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
    )

    loss = tf.reduce_mean(loss)
    return loss


# ==================================================================================================


def get_loss(predictions, samples):
    # Calculate logit length here, that we can decorate the loss calculation
    # with a tf-function call for much faster calculations
    logit_lengths = tf.constant(
        tf.shape(predictions)[0], shape=tf.shape(predictions)[1]
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

    # Drop all except first sample (shape is [n_steps, batch_size, n_hidden_6])
    predictions = tf.transpose(predictions, perm=[1, 0, 2])
    predictions = tf.expand_dims(predictions[0], axis=0)
    predictions = tf.transpose(predictions, perm=[1, 0, 2])

    logit_lengths = tf.constant(tf.shape(predictions)[0], shape=(1,))
    decoded = tf.nn.ctc_greedy_decoder(predictions, logit_lengths, merge_repeated=True)

    label = samples["label"][0]
    label = idx2char.lookup(label).numpy()
    label = b"".join(label).strip()
    print("Label: {}".format(label))

    values = tf.cast(decoded[0][0].values, dtype=tf.int32)
    values = idx2char.lookup(values).numpy()
    values = b"".join(values)
    print("Prediction: {}".format(values))


# ==================================================================================================


def train(dataset_train, dataset_val, start_epoch, stop_epoch):
    step = np.int64(0)
    # tf.profiler.experimental.start('/checkpoints/profiles/')

    for epoch in range(start_epoch, stop_epoch):
        start_time = time.time()

        for samples in dataset_train:

            # tf.summary.trace_on(graph=True, profiler=False)

            # Reset lstm states between batches, state is only required for streaming api
            if step != 0:
                model.reset_states()

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

            if step % 25 == 0:
                # print("")
                # print(samples["features"][0])
                # print(samples["label"][0])
                # print(predictions)
                log_greedy_text(predictions, samples)

        save_manager.save()
        eval(dataset_val)

        msg = "Epoch {} took {} hours\n"
        duration = utils.seconds_to_hours(time.time() - start_time)
        print(msg.format(epoch, duration))

    # tf.profiler.experimental.stop()


# ==================================================================================================


def eval(dataset_val):
    print("\nEvaluating ...")
    loss = 0
    step = 0
    for samples in dataset_val:
        features = samples["features"]
        predictions = model(features)
        loss += get_loss(predictions, samples).numpy()
        step += 1

        if step % 25 == 0:
            log_greedy_text(predictions, samples)

    loss = loss / step
    print("Validation loss: {}".format(loss))


# ==================================================================================================


def main():
    global model, summary_writer, save_manager, optimizer

    # Delete old data
    utils.delete_dir(checkpoint_dir)
    pipeline.delete_cache()

    summary_writer = tf.summary.create_file_writer(checkpoint_dir)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

    # tf.profiler.experimental.server.start(6009)
    # tf.summary.trace_on(graph=True, profiler=True)
    # tf.summary.trace_on(graph=True, profiler=False)

    model = nets.deepspeech.DeepSpeech(batch_size=batch_size)
    model.build(input_shape=(None, None, 26))
    model.summary()

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    save_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=3
    )

    start_epoch = 0
    if save_manager.latest_checkpoint:
        start_epoch = int(save_manager.latest_checkpoint.split("-")[-1])
        checkpoint.restore(save_manager.latest_checkpoint)
    start_epoch += 1

    dataset_train = pipeline.create_pipeline(
        dataset_train_path, batch_size, is_training=True
    )
    dataset_val = pipeline.create_pipeline(
        dataset_val_path, batch_size, is_training=False
    )
    train(dataset_train, dataset_val, start_epoch, max_epoch)
