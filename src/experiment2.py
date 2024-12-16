import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from keras.losses import BinaryCrossentropy, MeanSquaredError
from keras.metrics import BinaryAccuracy
import time
from misc import setup_experiment, setup_logger, now, setup_model, save_history
from data.load import get_data
from spektral.data import BatchLoader, DisjointLoader

if __name__ == '__main__':
    start_time = time.time()
    ######################################################################
    # SETUP
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = setup_experiment(sys.argv[1])
    logger = setup_logger(config['folder_path'])

    logger.info(f"Starting at {now()}")
    logger.info(f"Experiment saved in {config['folder_path']}")
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])
    ######################################################################

    # Load data and split it in train and test sets
    train_graphs, train_pairs, train_y, test_graphs, test_pairs, test_y = get_data(config)
    loader_tr = DisjointLoader(train_graphs, batch_size=config['batch_size'], epochs=config['epochs'])
    loader_te = DisjointLoader(test_graphs, batch_size=config['batch_size'], epochs=1)

    model = setup_model(config)
    # model.compile(optimizer=Adam(config['learning_rate']),
    #                              loss=BinaryCrossentropy(from_logits=True),
    #                              metrics=[BinaryAccuracy(threshold=.5)])
    optimizer = Adam(config['learning_rate'])
    loss_fn = MeanSquaredError()


    @tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
    def train_step(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(target, predictions) + sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss


    step = loss = 0
    for batch in loader_tr:
        step += 1
        loss += train_step(*batch)
        if step == loader_tr.steps_per_epoch:
            step = 0
            print("Loss: {}".format(loss / loader_tr.steps_per_epoch))
            loss = 0

    ################################################################################
    # Evaluate model
    ################################################################################
    print("Testing model")
    loss = 0
    for batch in loader_te:
        inputs, target = batch
        predictions = model(inputs, training=False)
        loss += loss_fn(target, predictions)
    loss /= loader_te.steps_per_epoch
    print("Done. Test loss: {}".format(loss))


    logger.info("--- %s seconds ---" % (time.time() - start_time))