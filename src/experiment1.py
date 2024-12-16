import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
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
    logger = setup_logger(config['folder_path'], config['logger']['level'], config['logger']['format'])

    logger.info(f"Starting at {now()}")
    logger.info(f"Experiment saved in {config['folder_path']}")
    np.random.seed(config['seed'])
    tf.random.set_seed(config['seed'])
    ######################################################################

    # Load data and split it in train and test sets
    train_graphs, train_pairs, train_y, test_graphs, test_pairs, test_y = get_data(config)
    #remove\/
    if config['pairwise']:
        logger.debug(type(train_graphs))
        logger.debug(train_graphs[0])
        logger.debug(type(train_pairs))
        logger.debug(train_pairs[0])
        logger.debug(type(train_y))
        logger.debug(train_y[0:31])
    #remove/\

    # Initialize the model, optimizer, and loss function
    model = setup_model(config)
    model.compile(optimizer=Adam(config['learning_rate']),
                                 loss=BinaryCrossentropy(from_logits=True),
                                 metrics=[BinaryAccuracy(threshold=.5)])

    ################################################################################
    # Fit model
    ################################################################################
    train_graphs_x = [graph.x for graph in train_graphs]
    train_graphs_a = [graph.a for graph in train_graphs]
    train_graphs_e = [graph.e for graph in train_graphs]
    train_data = list(zip(train_graphs_x, train_graphs_a, train_graphs_e))
    test_graphs_x = [graph.x for graph in test_graphs]
    test_graphs_a = [graph.a for graph in test_graphs]
    test_graphs_e = [graph.e for graph in test_graphs]
    test_data = list(zip(test_graphs_x, test_graphs_a, test_graphs_e))

    ragged_train_data = tf.ragged.constant(train_data)
    ragged_test_data = tf.ragged.constant(test_data)

    train_pair_a = [pair[0] for pair in train_pairs]
    train_pair_b = [pair[1] for pair in train_pairs]
    test_pair_a = [pair[0] for pair in test_pairs]
    test_pair_b = [pair[1] for pair in test_pairs]
    #remove\/

    if config['pairwise']:
        logger.debug(train_graphs[0].x.shape)
        logger.debug(train_data[0][0].shape)
        # logger.debug(ragged_train_data[0][0].shape)
        logger.debug(train_graphs[0].a.shape)
        logger.debug(train_data[0][1].shape)
        # logger.debug(ragged_train_data[0][1].shape)
        logger.debug(train_graphs[0].e.shape)
        logger.debug(train_data[0][2].shape)
        # logger.debug(ragged_train_data[0][2].shape)
    #remove/\

    train = tf.data.Dataset.from_tensors(((ragged_train_data, train_pair_a, train_pair_b), train_y))
    test = tf.data.Dataset.from_tensors(((ragged_test_data, test_pair_a, test_pair_b), test_y))

    h = model.fit(train,
                  validation_data=test,
                  batch_size=config['batch_size'],
                  epochs=config['epochs'],
                  verbose=2)

    save_history(h, config['folder_path'])
    logger.info(f"Training done!")
    ################################################################################


    logger.info("--- %s seconds ---" % (time.time() - start_time))