import json
import os
import shutil
import time

import tensorflow as tf
import tensorflow_addons as tfa

from . import nets, pipeline, utils

# from tensorflow.keras.mixed_precision import experimental as mixed_precision


# ==================================================================================================

# tf.config.run_functions_eagerly(True)
# tf.config.optimizer.set_jit(True)

config = utils.get_config()
checkpoint_dir = config["checkpoint_dir"]
cache_dir = config["cache_dir"]

alphabet = utils.load_alphabet(config)
idx2char: tf.lookup.StaticHashTable

model: tf.keras.Model
summary_writer: tf.summary.SummaryWriter
save_manager: tf.train.CheckpointManager
optimizer: tf.keras.optimizers.Adam
strategy: tf.distribute.Strategy

# ==================================================================================================


def create_idx2char():
    global idx2char
    idx2char = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant([i for i, u in enumerate(alphabet)]),
            values=tf.constant([u for i, u in enumerate(alphabet)]),
        ),
        default_value=tf.constant(" "),
    )


# ==================================================================================================


@tf.function(experimental_relax_shapes=True)
def get_loss(predictions, samples):
    """Calculate CTC loss"""

    label_lengths = samples["label_length"]
    labels = samples["label"]
    logit_lengths = samples["feature_length"] / model.get_time_reduction_factor()
    logit_lengths = tf.cast(tf.math.ceil(logit_lengths), tf.int32)

    # Blank index of "-1" returned better results compared to labels starting from 1, reason unclear
    loss = tf.nn.ctc_loss(
        labels=labels,
        logits=predictions,
        label_length=label_lengths,
        logit_length=logit_lengths,
        blank_index=-1,
        logits_time_major=False,
    )
    return loss


# ==================================================================================================


@tf.function(experimental_relax_shapes=True)
def train_step(samples):
    """Run a single forward and backward step and return the loss"""

    features = samples["features"]

    with tf.GradientTape() as tape:
        predictions = model(features)
        loss = get_loss(predictions, samples)
        loss = tf.reduce_mean(loss)

    trainable_variables = model.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    # gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    gradients, global_norm = tf.clip_by_global_norm(gradients, 1.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


# ==================================================================================================


@tf.function(experimental_relax_shapes=True)
def eval_step(samples):
    """Run a single forward and return the loss"""

    features = samples["features"]
    predictions = model(features, training=False)
    loss = get_loss(predictions, samples)
    loss = tf.reduce_mean(loss)

    return loss


# ==================================================================================================


@tf.function(experimental_relax_shapes=True)
def distributed_train_step(dist_inputs):
    """Helper function for distributed training"""

    per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
    loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    return loss


# ==================================================================================================


@tf.function(experimental_relax_shapes=True)
def distributed_eval_step(dist_inputs):
    """Helper function for distributed evaluation"""

    per_replica_losses = strategy.run(eval_step, args=(dist_inputs,))
    loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    return loss


# ==================================================================================================


def log_greedy_text(samples):
    """Run a prediction and log the predicted text"""

    features = tf.expand_dims(samples["features"][0], axis=0)
    logit_lengths = tf.expand_dims(samples["feature_length"][0], axis=0)
    logit_lengths = logit_lengths / model.get_time_reduction_factor()
    logit_lengths = tf.cast(tf.math.ceil(logit_lengths), tf.int32)

    with tf.device("/CPU:0"):
        prediction = model.predict(features)

    # Switch batch_size and time_steps before decoding
    predictions = tf.transpose(prediction, perm=[1, 0, 2])
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


def distributed_log_greedy(dist_inputs):
    """Helper function for distributed prediction logs. Because the dataset is distributed, we have
    to extract the data values of the first device before running a non distributed prediction"""

    samples = strategy.experimental_local_results(dist_inputs)[0]
    conv_samps = {
        "features": samples["features"].values[0],
        "feature_length": samples["feature_length"].values[0],
        "label": samples["label"].values[0],
    }
    log_greedy_text(conv_samps)


# ==================================================================================================


def train(dataset_train, dataset_eval, start_epoch, stop_epoch):
    step = 0
    best_eval_loss = float("inf")
    epochs_without_improvement = 0
    log_greedy_steps = config["log_prediction_steps"]
    last_save_time = time.time()
    training_start_time = time.time()
    training_epochs = 0
    # tf.profiler.experimental.start('/checkpoints/profiles/')

    for epoch in range(start_epoch, stop_epoch):
        start_time = time.time()
        print("\nStarting new training epoch ...")

        for samples in dataset_train:

            # tf.summary.trace_on(graph=True, profiler=False)

            # tf.profiler.experimental.client.trace('grpc://localhost:6009',
            #                                       checkpoint_dir, 2000)

            # with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
            #     # Not recommended, but couldn't get next dataset element here
            #     loss, predictions = train_step(samples, step)

            loss = distributed_train_step(samples)
            step += 1

            with summary_writer.as_default():
                tf.summary.experimental.set_step(step)
                tf.summary.scalar("loss", loss)

            print("Step: {} - Epoch: {} - Loss: {}".format(step, epoch, loss.numpy()))

            # with summary_writer.as_default():
            #     tf.summary.trace_export(
            #         name="my_func_trace",
            #         step=0)

            if log_greedy_steps != 0 and step % log_greedy_steps == 0:
                distributed_log_greedy(samples)

            if (time.time() - last_save_time) / 60 > config["autosave_every_min"]:
                save_manager.save()

        # Evaluate
        eval_loss = eval(dataset_eval)

        # Save new best model
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            tf.keras.models.save_model(model, checkpoint_dir, include_optimizer=False)
            print("Saved new best validating model")

        training_epochs += 1
        msg = "Epoch {} took {} hours\n"
        duration = utils.seconds_to_hours(time.time() - start_time)
        print(msg.format(epoch, duration))

        # Count epochs without improvement for early stopping and reducing learning rate on plateaus
        if eval_loss > best_eval_loss - config["esrp_min_delta"]:
            epochs_without_improvement += 1
        else:
            epochs_without_improvement = 0

        # Early stopping
        if (
            config["use_early_stopping"]
            and epochs_without_improvement == config["early_stopping_epochs"]
        ):
            msg = "Early stop triggered as the loss did not improve the last {} epochs"
            print(msg.format(epochs_without_improvement))
            break

        # Reduce learning rate on plateau. If the learning rate was reduced and there is still
        # no improvement, wait reduce_lr_plateau_epochs before the learning rate is reduced again
        if (
            config["use_lrp_reduction"]
            and epochs_without_improvement > 0
            and epochs_without_improvement % config["reduce_lr_plateau_epochs"] == 0
        ):
            # Reduce learning rate
            new_lr = optimizer.learning_rate * config["lr_plateau_reduction"]
            optimizer.learning_rate = new_lr
            msg = "Encountered a plateau, reducing learning rate to {}"
            print(msg.format(optimizer.learning_rate))

            # Reload checkpoint that we use the best_dev weights again
            print("Reloading model with best weights ...")
            best_model = tf.keras.models.load_model(checkpoint_dir)
            model.set_weights(best_model.get_weights())

    msg = "\nCompleted training after {} epochs with best evaluation loss of {:.4f} after {} hours"
    duration = utils.seconds_to_hours(time.time() - training_start_time)
    print(msg.format(training_epochs, best_eval_loss, duration))

    # tf.profiler.experimental.stop()


# ==================================================================================================


def eval(dataset_eval):
    print("\nEvaluating ...")
    loss = 0
    step = 0
    log_greedy_steps = config["log_prediction_steps"]

    for samples in dataset_eval:
        loss += distributed_eval_step(samples).numpy()
        step += 1

        if log_greedy_steps != 0 and step % log_greedy_steps == 0:
            distributed_log_greedy(samples)

    loss = loss / step
    print("Validation loss: {}".format(loss))
    return loss


# ==================================================================================================


def main():
    global model, summary_writer, save_manager, optimizer, strategy

    # Use growing gpu memory
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)

    # Build this after setting the gpu config, else it will raise an initialization error
    create_idx2char()

    if config["empty_cache_dir"]:
        # Delete and recreate cache dir
        if os.path.exists(cache_dir):
            utils.delete_dir(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    if config["empty_ckpt_dir"]:
        # Delete and recreate checkpoint dir
        if os.path.exists(checkpoint_dir):
            utils.delete_dir(checkpoint_dir)

    if config["continue_pretrained"]:
        # Copy the pretrained checkpoint
        shutil.copytree(config["pretrained_checkpoint_dir"], config["checkpoint_dir"])
        print("Copied pretrained checkpoint")
    else:
        # Create and empty directory
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Enable testing with multiple gpus
    strategy = tf.distribute.MirroredStrategy()
    global_train_batch_size = (
        config["batch_sizes"]["train"] * strategy.num_replicas_in_sync
    )
    global_eval_batch_size = (
        config["batch_sizes"]["eval"] * strategy.num_replicas_in_sync
    )

    # Create pipelines
    cache = config["cache_dir"] + "train" if config["use_pipeline_cache"] else ""
    dataset_train = pipeline.create_pipeline(
        csv_path=config["data_paths"]["train"],
        batch_size=global_train_batch_size,
        config=config,
        augment=True,
        cache_path=cache,
    )
    cache = config["cache_dir"] + "eval" if config["use_pipeline_cache"] else ""
    dataset_eval = pipeline.create_pipeline(
        csv_path=config["data_paths"]["eval"],
        batch_size=global_eval_batch_size,
        config=config,
        augment=False,
        cache_path=cache,
    )
    dataset_train = strategy.experimental_distribute_dataset(dataset_train)
    dataset_eval = strategy.experimental_distribute_dataset(dataset_eval)

    # tf.profiler.experimental.server.start(6009)
    # tf.summary.trace_on(graph=True, profiler=True)
    # tf.summary.trace_on(graph=True, profiler=False)

    feature_type = config["audio_features"]["use_type"]
    c_input = config["audio_features"][feature_type]["num_features"]
    c_output = len(alphabet) + 1

    # Get the model type either from the config or the existing checkpoint
    if config["continue_pretrained"] or not config["empty_ckpt_dir"]:
        path = os.path.join(checkpoint_dir, "config_export.json")
        exported_config = utils.load_json_file(path)
        network_type = exported_config["network"]["name"]
    else:
        network_type = config["network"]["name"]

    # Build the network
    print("Creating new {} model ...".format(network_type))
    with strategy.scope():
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

    # Copy weights. Compared to loading the model directly, this has the benefit that parts of the
    # model code can be changed as long the layers are kept.
    if config["continue_pretrained"] or not config["empty_ckpt_dir"]:
        print("Copying model weights from checkpoint ...")
        exported_model = tf.keras.models.load_model(checkpoint_dir)
        model.set_weights(exported_model.get_weights())

    # Print network summary
    model.build(input_shape=(None, None, c_input))
    model.summary()
    # tf.keras.models.save_model(model, checkpoint_dir, include_optimizer=False)

    # Export current config next to the checkpoints
    path = os.path.join(checkpoint_dir, "config_export.json")
    with open(path, "w+", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    # Select optimizer
    with strategy.scope():
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
                beta_1=config["optimizer"]["beta1"],
                beta_2=config["optimizer"]["beta2"],
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
        # checkpoint.restore(save_manager.latest_checkpoint)
    start_epoch += 1

    # Finally the training can start
    max_epoch = config["training_epochs"]
    train(dataset_train, dataset_eval, start_epoch, max_epoch)
