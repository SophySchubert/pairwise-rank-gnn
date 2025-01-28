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

from tensorflow.keras.layers import Dense
from spektral.layers import ECCConv

from misc import setup_experiment, setup_logger, now, setup_model
from data.load import get_data
from data.loader import MyDisjointLoader

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
    train_graphs, test_graphs, base_ranking = get_data(config)
    loader_tr = MyDisjointLoader(train_graphs, batch_size=config['batch_size'], epochs=config['epochs'], seed=config['seed'])
    loader_te = MyDisjointLoader(test_graphs, batch_size=config['batch_size'], epochs=1, seed=config['seed'])

    #########
    # model #
    #########
    def pref_lookup(X, pref_a, pref_b):
        X_a = tf.gather(X, pref_a, axis=0)
        X_b = tf.gather(X, pref_b, axis=0)
        return X_a, X_b

    def createPairwiseModel(config, inputs):
        inputs = tf.keras.Input(shape=(None, None), name='inputs')
        x, a, e, i, idx_a, idx_b = inputs#ohne batch dimension None am Anfang, anders als beim DataLoader, Input statt InputLayer

        x = tf.cast(x, tf.float32)
        a = a.with_values(tf.cast(a.values, tf.float32))
        e = tf.cast(e, tf.float32)

        conv1 = ECCConv(32, activation="relu")([x, a, e])
        conv2 = ECCConv(32, activation="relu")([conv1, a, e])
        x_util = Dense(config['n_out'], activation=None)(conv2)
        X_a, X_b = pref_lookup(x_util, idx_a, idx_b)
        out = X_b - X_a

        m = tf.keras.Model(inputs=[x, a, e, idx_a, idx_b], outputs=out, name="RankNet")
        m_infer = tf.keras.Model(inputs=[x, a, e], outputs=x_util, name="RankNet_predictor")
        return m, m_infer

    # model = createPairwiseModel(config)
    model = setup_model(config)
    model.compile(optimizer=Adam(config['learning_rate']),
                  loss=BinaryCrossentropy(from_logits=True),
                  metrics=[BinaryAccuracy(threshold=.5)])


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

