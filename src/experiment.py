import sys
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy
import time
from misc import setup_experiment, setup_logger, now, setup_model, save_history
from data.load import get_data
from spektral.data import BatchLoader

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
    print(type(train_graphs))
    print(train_graphs[0])
    print(type(train_pairs))
    print(train_pairs[0])
    print(type(train_y))
    print(train_y[0])

    # Initialize the model, optimizer, and loss function
    model = setup_model(config)
    model.compile(optimizer=Adam(config['learning_rate']),
                                 loss=BinaryCrossentropy(from_logits=True),
                                 metrics=[BinaryAccuracy(threshold=.0)])

    ################################################################################
    # Fit model
    ################################################################################
#     loader_train = BatchLoader((train_graphs, train_pairs, train_y), batch_size=config['batch_size'], mask=True)
#     h = model.fit(loader_train.load(), steps_per_epoch=loader_train.steps_per_epoch, epochs=config['epochs'], verbose=2)

    train_graphs_x = np.array([graph.x for graph in train_graphs])
    train_graphs_a = np.array([graph.a for graph in train_graphs])
    train_graphs_e = np.array([graph.e for graph in train_graphs])
    train_data = list(zip(train_graphs_x, train_graphs_a, train_graphs_e))
    train = tf.data.Dataset.from_tensors((train_data, train_pairs, train_y))
    h = model.fit(train, batch_size=config['batch_size'], epochs=config['epochs'], verbose=2)
    save_history(h, config['folder_path'])
    logger.info(f"Training done!")
    ################################################################################


    logger.info("--- %s seconds ---" % (time.time() - start_time))