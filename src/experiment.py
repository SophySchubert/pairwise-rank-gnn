import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from misc import setup_experiment, setup_logger, now
from data.load import split_data
from models.general_gnn import GeneralGNN
from models.prgnn import PRGNN


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = setup_experiment(sys.argv[1])

    logger = setup_logger(config['folder_path'])

    logger.info(f"Starting at {now()}")
    logger.info(f"Experiment saved in {config['folder_path']}")
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])

    # Load data and split it in train and test sets
    loader_train, loader_test, n_labels = split_data(config)

    # Initialize the model, optimizer, and loss function
    model = None
    if config['model'] == 'general_gnn':
        model = GeneralGNN(n_labels, activation="softmax")
    elif config['model'] == 'prgnn':
        model = PRGNN(hidden=config['hidden_layers'], config=config)
    else:
        raise ValueError(f"Unknown model: {config['model']}")
    optimizer = Adam(learning_rate=config['learning_rate'])
    loss_function = CategoricalCrossentropy()


    ################################################################################
    # Fit model
    ################################################################################
    @tf.function(input_signature=loader_train.tf_signature(), experimental_relax_shapes=True)
    def train_step(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_function(target, predictions) + sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        acc = tf.reduce_mean(categorical_accuracy(target, predictions))
        return loss, acc

    @tf.function(input_signature=loader_train.tf_signature(), experimental_relax_shapes=True)
    def train_step_pairs(inputs1, inputs2, target):
        with tf.GradientTape() as tape:
            predictions = model([inputs1, inputs2], training=True)
            loss = loss_function(target, predictions) + sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        acc = tf.reduce_mean(categorical_accuracy(target, predictions))
        return loss, acc

    def evaluate(loader):
        output = []
        step = 0
        while step < loader.steps_per_epoch:
            step += 1
            inputs, target = loader.__next__()
            pred = model(inputs, training=False)
            outs = (
                loss_function(target, pred),
                tf.reduce_mean(categorical_accuracy(target, pred)),
                len(target),  # Keep track of batch size
            )
            output.append(outs)
            if step == loader.steps_per_epoch:
                output = np.array(output)
                return np.average(output[:, :-1], 0, weights=output[:, -1])

    def evaluate_pairs(loader):
        output = []
        step = 0
        while step < loader.steps_per_epoch:
            step += 1
            inputs1, inputs2, target = loader.__next__()
            pred = model([inputs1, inputs2], training=False)
            outs = (
                loss_function(target, pred),
                tf.reduce_mean(categorical_accuracy(target, pred)),
                len(target),  # Keep track of batch size
            )
            output.append(outs)
            if step == loader.steps_per_epoch:
                output = np.array(output)
                return np.average(output[:, :-1], 0, weights=output[:, -1])

    epoch = step = 0
    results = []
    for batch in loader_train:
        step += 1
        loss, acc = 0, 0
        if config['pairwise'] == True:
            inputs1, inputs2, target = batch
            loss, acc = train_step_pairs(inputs1, inputs2, target)
        else:
            loss, acc = train_step(*batch)
        results.append((loss, acc))
        if step == loader_train.steps_per_epoch:
            step = 0
            epoch += 1
            results_test = []
            if config['pairwise'] == True:
                results_test = evaluate_pairs(loader_test)
            else:
                results_test = evaluate(loader_test)
            logger.info(
                "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Test loss: {:.3f} - Test acc: {:.3f}".format(
                    epoch, *np.mean(results, 0), *results_test
                )
            )
            results = []

    ################################################################################
    # Evaluate model
    ################################################################################
    results_test = []
    if config['pairwise'] == True:
        results_test = evaluate_pairs(loader_test)
    else:
        results_test = evaluate(loader_test)
    logger.info("Final results - Loss: {:.3f} - Acc: {:.3f}".format(*results_test))



