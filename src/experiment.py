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
    train_data, test_data = get_data(config)
    train_y = [(tuple[0].y, tuple[1].y) for tuple in train_data]
    test_y = [(tuple[0].y, tuple[1].y) for tuple in test_data]

    # Initialize the model, optimizer, and loss function
    model = setup_model(config)
    model.compile(optimizer=Adam(config['learning_rate']),
                                 loss=BinaryCrossentropy(from_logits=True),
                                 metrics=[BinaryAccuracy(threshold=.0)])

    ################################################################################
    # Fit model
    ################################################################################
    # loader_train = BatchLoader(train_data, batch_size=config['batch_size'], mask=True)
    # h = model.fit(loader_train.load(), steps_per_epoch=loader_train.steps_per_epoch, epochs=config['epochs'], verbose=2)
    h = model.fit(train_data, train_y, batch_size=config['batch_size'], epochs=config['epochs'], verbose=2)
    save_history(h, config['folder_path'])
    logger.info(f"Training done!")
    ################################################################################


    logger.info("--- %s seconds ---" % (time.time() - start_time))