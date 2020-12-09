import os
import shutil
import time

import tensorflow as tf

from . import nets, pipeline, utils

# ==================================================================================================

physical_devices = tf.config.list_physical_devices("GPU")
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

model: tf.keras.Model
summary_writer: tf.summary.SummaryWriter
save_manager: tf.train.CheckpointManager
optimizer: tf.keras.optimizers.Adam

batch_size = 16
max_epoch = 30
learning_rate = 0.001
# dataset_train_path = "/data_prepared/de/voxforge/train_azce.csv"
dataset_train_path = "/data_prepared/de/voxforge/mini_test_azce.csv"
# dataset_train_path = "/data_prepared/de/voxforge/nano_test_azce.csv"
dataset_val_path = "/data_prepared/de/voxforge/test_azce.csv"
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


def delete_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


# ==================================================================================================


@tf.function
def loss_function(predictions, samples):
    labels = samples["label"]
    label_lengths = samples["label_length"]
    logit_lengths = samples["feature_length"]
    # logit_lengths = [38]

    # Blank index of "-1" did return better results, reason unclear
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

# @tf.function
def train_step(samples, step):
    features = samples["features"]

    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = loss_function(predictions, samples)

    with summary_writer.as_default():
        tf.summary.experimental.set_step(step)
        tf.summary.scalar("loss", loss)

    trainable_variables = model.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    # gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, predictions


# ==================================================================================================


def log_greedy_text(predictions, samples):

    # Drop all except first sample (shape is [n_steps, batch_size, n_hidden_6])
    predictions = tf.transpose(predictions, perm=[1, 0, 2])
    predictions = tf.expand_dims(predictions[0], axis=0)
    predictions = tf.transpose(predictions, perm=[1, 0, 2])

    logit_lengths = [samples["feature_length"][0]]
    # logit_lengths = [38]
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


def train(dataset, start_epoch, stop_epoch):
    step = 0

    for epoch in range(start_epoch, stop_epoch):
        start_time = time.time()

        for samples in dataset:
            # Reset lstm states between batches, state is only required for streaming api
            if step != 0:
                model.reset_states()

            loss, predictions = train_step(samples, step)
            step += 1

            print("Step: {} - Epoch: {} - Loss: {}".format(step, epoch, loss.numpy()))

            if step % 5 == 0:
                # print("")
                # print(samples["features"][0])
                # print(samples["label"][0])
                # print(predictions)
                log_greedy_text(predictions, samples)

        save_manager.save()

        msg = "Epoch {} took {} hours\n"
        duration = utils.seconds_to_hours(time.time() - start_time)
        print(msg.format(epoch, duration))


# ==================================================================================================


def main():
    global model, summary_writer, save_manager, optimizer

    model = nets.deepspeech.DeepSpeech(batch_size=batch_size)
    # model.build(input_shape=(16, 132, 26))
    # model.summary()

    delete_dir(checkpoint_dir)

    summary_writer = tf.summary.create_file_writer(checkpoint_dir)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    save_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=3
    )

    start_epoch = 0
    if save_manager.latest_checkpoint:
        start_epoch = int(save_manager.latest_checkpoint.split("-")[-1])
        checkpoint.restore(save_manager.latest_checkpoint)
    start_epoch += 1

    pipeline.delete_cache()
    dataset = pipeline.create_pipeline(dataset_train_path, batch_size, is_training=True)
    train(dataset, start_epoch, max_epoch)
