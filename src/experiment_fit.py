import sys
import time
import shutil
import numpy as np
import tensorflow as tf
import pandas as pd
import spektral
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
    shutil.copy('src/models/prgnn.py', config['folder_path']+'/model.py')
    ######################################################################

    # Load data and split it in train and test sets
    train_graphs, test_graphs = get_data(config)
    loader_tr = MyDisjointLoader(train_graphs, batch_size=config['batch_size'], epochs=config['epochs'], seed=config['seed'])
    loader_te = MyDisjointLoader(test_graphs, batch_size=config['batch_size'], epochs=1, seed=config['seed'], radius=1, sampling_ratio=1)

    #########
    # model #
    #########
    def pref_lookup(X, pref_a, pref_b):
        X_a = tf.gather(X, pref_a, axis=0)
        X_b = tf.gather(X, pref_b, axis=0)
        return X_a, X_b

    def myModel():
        x = tf.keras.Input(shape=(None, None, 9), dtype=tf.int64)
        a = tf.keras.Input(shape=(None, None, None), dtype=tf.float64, sparse=True)
        e = tf.keras.Input(shape=(None, None, 3),dtype=tf.int64)
        i = tf.keras.Input(shape=(None, None,), dtype=tf.int64)
        pref_a = tf.keras.Input(shape=(None, 12800,), dtype=tf.int64)
        pref_b = tf.keras.Input(shape=(None, 12800,), dtype=tf.int64)
        x_0 = tf.cast(x, tf.float32)
        

        x_1 = spektral.layers.GCNConv(128, activation="relu")([x_0, a])
        x_1 = tf.keras.layers.BatchNormalization()(x_1)
        x_1 = tf.keras.layers.Dropout(0.5)(x_1)
        x_2 = spektral.layers.GCNConv(128, activation="relu")([x_1, a])
        x_2 = tf.keras.layers.BatchNormalization()(x_2)
        x_2 = tf.keras.layers.Dropout(0.5)(x_2)
        X_utils = spektral.layers.GCNConv(1, activation="softmax")([x_2, a])

        # X_utils = keras.layers.Dense(1, activation="relu")(x_3)
        X_a, X_b = pref_lookup(X_utils, pref_a, pref_b)
        out = X_b - X_a

        return out

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
    ################################################################################
    # Evaluate model
    ################################################################################
    logger.info("Testing model")
    loss, acc = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
    logger.info(f"Done. Test loss: {loss} - Test Accuracy: {acc}")


    logger.info("--- %s seconds ---" % (time.time() - start_time))
    ###############################################################################
    df = pd.DataFrame({'loss': hs.history['loss'], 'binary_accuracy': hs.history['binary_accuracy']})
    df.to_csv(config['folder_path'] + '/loss_acc.csv', index=False)

