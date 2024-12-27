import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from keras.losses import BinaryCrossentropy, MeanSquaredError
from keras.metrics import BinaryAccuracy

from misc import setup_experiment, setup_logger, now, setup_model
from data.load import get_data
from data.misc import MyDisjointLoader

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
    loader_tr = MyDisjointLoader(train_graphs, batch_size=config['batch_size'], epochs=config['epochs'], seed=config['seed'])
    loader_te = MyDisjointLoader(test_graphs, batch_size=config['batch_size'], epochs=1, seed=config['seed'])

    model = setup_model(config)
    model.compile(optimizer=Adam(config['learning_rate']),
                                 loss=BinaryCrossentropy(from_logits=True),
                                 metrics=[BinaryAccuracy(threshold=.5)])
    optimizer = Adam(config['learning_rate'])
    loss_fn = MeanSquaredError()


    ################################################################################
    # Fit model
    ################################################################################
    @tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
    def train_step(inputs, target):
        print("train_step")
        # print(f"len(inputs): {len(inputs)}")
        # print(f"inputs: {inputs}")
        # print(f"x: {inputs[0][0]}")
        # print(f"a: {inputs[0][1]}")
        # print(f"e: {inputs[0][2]}")
        # print(f"i: {inputs[0][3]}")
        # print(f"idx_a: {inputs[1]}")
        # print(f"idx_b: {inputs[2]}")
        # print(f"target: {target}")
        # print("train_step")
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(target, predictions) + sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss


    step = loss = 0
    for batch in loader_tr:#batch[0]==axei, batch[1]==idx_a, batch[2]==idx_b, batch[3]==target
        step += 1
        print("batch in loader_tr")
        # print(f"len(batch): {len(batch)}")
        # print(f"len(inputs): {len(batch[0])}")
        # print(f"inputs: {batch[0]}")
        # print(f"x: {batch[0][0][0]}")
        # print(f"a: {batch[0][0][1]}")
        # print(f"e: {batch[0][0][2]}")
        # print(f"i: {batch[0][0][3]}")
        # print(f"idx_a: {batch[0][1]}")
        # print(f"idx_b: {batch[0][2]}")
        # print(f"target: {batch[1]}")
        # print("batch in loader_tr")
        # print(loader_tr.tf_signature())

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