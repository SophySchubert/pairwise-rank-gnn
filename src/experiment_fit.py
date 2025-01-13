import sys
import time
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kendalltau
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
    train_graphs, test_graphs = get_data(config)
    loader_tr = MyDisjointLoader(train_graphs, batch_size=config['batch_size'], epochs=config['epochs'], seed=config['seed'])
    loader_te = MyDisjointLoader(test_graphs, batch_size=config['batch_size'], epochs=1, seed=config['seed'], radius=1, sampling_ratio=1)

    model = setup_model(config)
    model.compile(optimizer=Adam(config['learning_rate']),
                                 loss=BinaryCrossentropy(from_logits=True),
                                 metrics=[BinaryAccuracy(threshold=.0)])
    optimizer = Adam(config['learning_rate'])
    loss_fn = MeanSquaredError()
    accuracy_fn = BinaryAccuracy(threshold=.0)


    ################################################################################
    # Fit model
    ################################################################################

    hs = model.fit(loader_tr.load(), steps_per_epoch=loader_tr.steps_per_epoch, epochs=config['epochs'], verbose=1)
    df = pd.DataFrame({'loss': hs.history['loss'], 'binary_accuracy': hs.history['binary_accuracy']})
    df.to_csv(config['folder_path'] + '/loss_acc.csv', index=False)
    ################################################################################
    # Evaluate model
    ################################################################################
    logger.info("Testing model")
    loss = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
    logger.info("Done. Test loss: {}".format(loss))


    logger.info("--- %s seconds ---" % (time.time() - start_time))
    ###############################################################################

