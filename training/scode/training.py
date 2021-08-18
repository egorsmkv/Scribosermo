import json
import os
import shutil
import time

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision

# Required for loading exported models with tf=2.3, which is required for tflite exports
from tensorflow.python.framework.errors_impl import (  # pylint: disable=no-name-in-module
    NotFoundError,
)

from . import nets, pipeline, utils

# ==================================================================================================

# tf.config.run_functions_eagerly(True)
# tf.config.optimizer.set_jit(True)

config = utils.get_config()
checkpoint_dir = config["checkpoint_dir"]

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
        predictions = model(features, training=True)
        loss = get_loss(predictions, samples)
        loss = tf.reduce_mean(loss)

        if config["mixed_precision"]:
            scaled_loss = optimizer.get_scaled_loss(loss)

    trainable_variables = model.trainable_variables
    if config["freeze_base_net"]:
        trainable_variables = model.trainable_variables[-2:]
        print("Training only last layer")

    if config["mixed_precision"]:
        gradients = tape.gradient(scaled_loss, trainable_variables)
        gradients = optimizer.get_unscaled_gradients(gradients)
    else:
        gradients = tape.gradient(loss, trainable_variables)

    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

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


def log_greedy_text(samples, trainmode=True):
    """Run a prediction and log the predicted text"""
    global idx2char

    features = tf.expand_dims(samples["features"][0], axis=0)
    logit_lengths = tf.expand_dims(samples["feature_length"][0], axis=0)
    logit_lengths = logit_lengths / model.get_time_reduction_factor()
    logit_lengths = tf.cast(tf.math.ceil(logit_lengths), tf.int32)

    if trainmode:
        # Using model.predict() instead didn't work here
        prediction = model(features, training=False)
    else:
        prediction = model.predict(features)

    # Switch batch_size and time_steps before decoding
    predictions = tf.transpose(prediction, perm=[1, 0, 2])
    decoded = tf.nn.ctc_greedy_decoder(predictions, logit_lengths, merge_repeated=True)

    label = samples["label"][0]
    label = idx2char.lookup(label).numpy()
    label = b"".join(label).strip().decode("utf-8")
    print("=Label======: {}".format(label))

    values = tf.cast(decoded[0][0].values, dtype=tf.int32)
    values = idx2char.lookup(values).numpy()
    values = b"".join(values).decode("utf-8")
    print("=Prediction=: {}".format(values))


# ==================================================================================================


def distributed_log_greedy(dist_inputs):
    """Helper function for distributed prediction logs. Because the dataset is distributed, we have
    to extract the data values of the first device before running a non distributed prediction"""

    samples = strategy.experimental_local_results(dist_inputs)[0]

    if hasattr(samples["features"], "values"):
        # Multi-GPU distribution
        conv_samps = {
            "features": samples["features"].values[0],
            "feature_length": samples["feature_length"].values[0],
            "label": samples["label"].values[0],
        }
    else:
        # Single-GPU distribution
        conv_samps = {
            "features": samples["features"],
            "feature_length": samples["feature_length"],
            "label": samples["label"],
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

    if config["eval_before_start"]:
        eval_loss = evaluate(dataset_eval)
        best_eval_loss = eval_loss

    for epoch in range(start_epoch, stop_epoch):
        start_time = time.time()
        epoch_steps = 0
        print("\nStarting new training epoch ...")

        dist_dataset_iterator = iter(dataset_train)
        for samples in dist_dataset_iterator:

            if epoch_steps in config["profile_steps"]:
                # Train step with profiling
                with tf.profiler.experimental.Profile(checkpoint_dir):
                    with tf.profiler.experimental.Trace("train", step_num=step, _r=1):
                        print("Profiling performance of next step ...")
                        samples = next(dist_dataset_iterator)
                        loss = distributed_train_step(samples)
            else:
                # Normal train step
                loss = distributed_train_step(samples)

            step += 1
            epoch_steps += 1
            print("Step: {} - Epoch: {} - Loss: {}".format(step, epoch, loss.numpy()))

            with summary_writer.as_default():
                tf.summary.experimental.set_step(step)
                tf.summary.scalar("loss", loss)

            if log_greedy_steps != 0 and step % log_greedy_steps == 0:
                distributed_log_greedy(samples)

            if (time.time() - last_save_time) / 60 > config["autosave_every_min"]:
                save_manager.save()
                last_save_time = time.time()

        # Evaluate
        eval_loss = evaluate(dataset_eval)

        # Count epochs without improvement for early stopping and reducing learning rate on plateaus
        if eval_loss > best_eval_loss - config["esrp_min_delta"]:
            epochs_without_improvement += 1
        else:
            epochs_without_improvement = 0

        # Save new best model
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            model.save_weights(checkpoint_dir)
            print("Saved new best validating model")

        training_epochs += 1
        msg = "Epoch {} took {} hours\n"
        duration = utils.seconds_to_hours(time.time() - start_time)
        print(msg.format(epoch, duration))

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
            model.load_weights(checkpoint_dir)

    msg = "\nCompleted training after {} epochs with best evaluation loss of {:.4f} after {} hours"
    duration = utils.seconds_to_hours(time.time() - training_start_time)
    print(msg.format(training_epochs, best_eval_loss, duration))


# ==================================================================================================


def evaluate(dataset_eval):
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


def build_pipelines():
    """Initialize train/eval data pipelines"""

    global_train_batch_size = (
        config["batch_sizes"]["train"] * strategy.num_replicas_in_sync
    )
    global_eval_batch_size = (
        config["batch_sizes"]["eval"] * strategy.num_replicas_in_sync
    )

    # Create pipelines
    dataset_train = pipeline.create_pipeline(
        csv_path=config["data_paths"]["train"],
        batch_size=global_train_batch_size,
        config=config,
        mode="train",
    )
    dataset_eval = pipeline.create_pipeline(
        csv_path=config["data_paths"]["eval"],
        batch_size=global_eval_batch_size,
        config=config,
        mode="eval",
    )

    # Solve "Found an unshardable source dataset" warning, run before distribution
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.DATA
    )
    dataset_train = dataset_train.with_options(options)
    dataset_eval = dataset_eval.with_options(options)

    # Distribute datasets
    dataset_train = strategy.experimental_distribute_dataset(dataset_train)
    dataset_eval = strategy.experimental_distribute_dataset(dataset_eval)

    return dataset_train, dataset_eval


# ==================================================================================================


def create_optimizer():
    """Initialize training optimizer"""

    with strategy.scope():
        optimizer_type = config["optimizer"]["name"]
        if optimizer_type == "adamw":
            optim = tfa.optimizers.AdamW(
                learning_rate=config["optimizer"]["learning_rate"],
                weight_decay=config["optimizer"]["weight_decay"],
            )
        elif optimizer_type == "novograd":
            optim = tfa.optimizers.NovoGrad(
                learning_rate=config["optimizer"]["learning_rate"],
                weight_decay=config["optimizer"]["weight_decay"],
                beta_1=config["optimizer"]["beta1"],
                beta_2=config["optimizer"]["beta2"],
            )

        if config["mixed_precision"]:
            optim = mixed_precision.LossScaleOptimizer(optim)

    return optim


# ==================================================================================================


def build_new_model(
    new_config: dict, using_config_export: bool = False, print_log: bool = True
):
    feature_type = new_config["audio_features"]["use_type"]
    c_input = new_config["audio_features"][feature_type]["num_features"]
    c_output = len(utils.load_alphabet(new_config)) + 1

    # Get the model type either from the given config file (else-case)
    # or the from the model the training is continued on (if-case)
    if not using_config_export and (
        new_config["continue_pretrained"] or not new_config["empty_ckpt_dir"]
    ):
        path = os.path.join(checkpoint_dir, "config_export.json")
        exported_config = utils.load_json_file(path)
        network_type = exported_config["network"]["name"]
    else:
        network_type = new_config["network"]["name"]

    if print_log:
        print("Creating new {} model ...".format(network_type))

    # Create the network
    mynet = getattr(nets, network_type)
    new_model = mynet.MyModel(
        c_input=c_input, c_output=c_output, netconfig=new_config["network"]
    )

    new_model.build(input_shape=(None, None, c_input))
    new_model.compile()
    return new_model


# ==================================================================================================


def copy_weights(exported_model, new_model) -> None:
    """Copy model weights. Compared to loading the model directly, this has the benefit that parts
    of the model code can be changed as long the layers are kept."""

    # Get shapes of last (decoding) layer
    last_layer_shape_exp = [w.shape for w in exported_model.get_weights()][-2]
    last_layer_shape_new = [w.shape for w in new_model.get_weights()][-2]

    if last_layer_shape_new == last_layer_shape_exp:
        # Copy all weights
        new_model.set_weights(exported_model.get_weights())
        print("Copied weights of all layers")
    else:
        # Copy exported weights from all but the last layer.
        merged_weights = exported_model.get_weights()[:-2]

        if config["extend_old_alphabet"]:
            # Use some parts of the exported weights and some from the newly initialized
            print("Extending last layer ...")
            ell_weights = exported_model.get_weights()[-2:]
            nll_weights = new_model.get_weights()[-2:]

            for ell, nll in zip(ell_weights, nll_weights):
                # Insert the new character between the old ones and the ctc-symbol
                first_chars = ell[..., : ell.shape[-1] - 1]
                new_chars = nll[..., ell.shape[-1] - 1 : nll.shape[-1] - 1]
                ctc_char = ell[..., -1:]

                expanded = tf.concat([first_chars, new_chars, ctc_char], axis=-1)
                merged_weights.append(expanded)

        elif config["shrink_old_alphabet"]:
            # Drop a few alphabet characters at the end
            print("Shrinking last layer ...")
            ell_weights = exported_model.get_weights()[-2:]
            nll_weights = new_model.get_weights()[-2:]

            for ell, nll in zip(ell_weights, nll_weights):
                # Drop characters directly before the ctc-symbol
                first_chars = ell[..., : nll.shape[-1] - 1]
                ctc_char = ell[..., -1:]

                reduced = tf.concat([first_chars, ctc_char], axis=-1)
                merged_weights.append(reduced)

        else:
            # Keep the newly initialized weights for the missing layer.
            print("Newly initializing last layer ...")
            nll_weights = new_model.get_weights()[-2:]
            merged_weights.extend(nll_weights)

        new_model.set_weights(merged_weights)


# ==================================================================================================


def load_exported_model(exported_dir: str):
    """Rebuild model and load weights, because exporting the full model and loading it again didn't
    work for some models due to problems with the input-signature for @tf.function decorator"""

    try:
        print("Trying to load weights directly ...")
        path = os.path.join(exported_dir, "config_export.json")
        exported_config = utils.load_json_file(path)
        exported_model = build_new_model(
            exported_config, using_config_export=True, print_log=False
        )
        exported_model.load_weights(exported_dir)
    except (OSError, NotFoundError):
        # Load old or exported models where not only the weights were saved
        print("Loading weights from exported model instead ...")
        exported_model = tf.keras.models.load_model(exported_dir)

    return exported_model


# ==================================================================================================


def main():
    global model, summary_writer, save_manager, optimizer, strategy

    print("Starting training with config:")
    print(json.dumps(config, indent=2))

    # Use growing gpu memory
    gpus = tf.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if config["mixed_precision"]:
        # Enable mixed precision training
        mixed_precision.set_global_policy("mixed_float16")
        print("Training with mixed precision ...")

    # Build this after setting the gpu config, else it will raise an initialization error
    create_idx2char()

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

    # Enable training with multiple gpus
    strategy = tf.distribute.MirroredStrategy()

    # Initialize data pipelines
    dataset_train, dataset_eval = build_pipelines()

    # Create and initialize the model
    with strategy.scope():
        model = build_new_model(config)

    # Optionally load exported weights
    if config["continue_pretrained"] or not config["empty_ckpt_dir"]:
        print("Copying model weights from existing checkpoint ...")
        exported_model = load_exported_model(checkpoint_dir)
        copy_weights(exported_model, model)

    # Select optimizer
    optimizer = create_optimizer()

    # Export current config next to the checkpoints
    path = os.path.join(checkpoint_dir, "config_export.json")
    with open(path, "w+", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    # Initialize checkpoint manager for intermediate model backups
    summary_writer = tf.summary.create_file_writer(checkpoint_dir)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    save_manager = tf.train.CheckpointManager(
        checkpoint, directory=checkpoint_dir, max_to_keep=1
    )

    # Optionally overwrite model with backup checkpoint
    if config["restore_ckpt_insteadof_pb_file"]:
        print("Overwriting model with backup from the ckpt file ...")
        with strategy.scope():
            checkpoint.restore(save_manager.latest_checkpoint)

    # Print model summary
    model.summary()
    img_path = os.path.join(config["checkpoint_dir"], "model.png")
    tf.keras.utils.plot_model(
        model.model, to_file=img_path, show_shapes=True, expand_nested=True
    )

    # Optionally save model before doing any training updates
    if config["save_fresh_model"]:
        model.save_weights(checkpoint_dir)
        print("Saved fresh model")

    # Finally the training can start
    start_epoch = 1
    max_epoch = config["training_epochs"]
    train(dataset_train, dataset_eval, start_epoch, max_epoch)
