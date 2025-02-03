import sys
import time
import shutil
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.optimizers import Adam
from keras.losses import BinaryCrossentropy, MeanSquaredError
from keras.metrics import BinaryAccuracy
from itertools import combinations

from misc import setup_experiment, setup_logger, now, setup_model
from data.load import get_data
from data.loader import MyDisjointLoader, CustomDataLoader
from data.misc import CustomDisjointedLoader, sample_preference_pairs2

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
    print(f"len train_graphs:{len(train_graphs)}")
    # exit(1)
    # loader_tr = MyDisjointLoader(train_graphs, batch_size=config['batch_size'], epochs=config['epochs'], seed=config['seed'])
    # loader_te = MyDisjointLoader(test_graphs, batch_size=config['batch_size'], epochs=1, seed=config['seed'])
    ##############setup C#####################

    def sample_preference_pairs(graphs):
        c = [(a, b, check_util(graphs, a,b)) for a, b in combinations(range(len(graphs)), 2)]
        idx_a = []
        idx_b = []
        target = []
        for id_a, id_b, t in c:
            idx_a.append(id_a)
            idx_b.append(id_b)
            target.append(t)
        return np.array(list(zip(idx_a,idx_b))), np.array(target).reshape(-1)

    def check_util(data, index_a, index_b):
        a = data[index_a]
        b = data[index_b]
        util_a = a.y
        util_b = b.y
        if util_a >= util_b:
            return 1
        else:
            return 0

    pairs, targets = sample_preference_pairs(train_graphs)

    # hat den [Op:GatherV2] Fehler
    data_loader = CustomDataLoader(train_graphs, pairs, targets, batch_size=32, seed=42)
    ######## setup D############
    # hat den [Op:GatherV2] Fehler

    pairs_and_targets_train = sample_preference_pairs2(train_graphs)
    print(f"pairs_and_targets:{pairs_and_targets_train}")
    data_loader_train = CustomDisjointedLoader(train_graphs, pairs_and_targets_train, config, node_level=False, batch_size=config['batch_size'], epochs=config['epochs'], shuffle=True)

    pairs_and_targets_test = sample_preference_pairs2(test_graphs)
    data_loader_test = CustomDisjointedLoader(train_graphs, pairs_and_targets_test, config, node_level=False, batch_size=config['batch_size'], epochs=config['epochs'], shuffle=True)
    #########
    # model #
    #########

    def combine_model(model):
        from tensorflow.keras.layers import Subtract, Activation
        # Extract the input and output from the given model
        X_input = model()
        X_util = model.output

        # Define the additional layers
        idx_a = tf.keras.Input(shape=(None,), name='idx_a')
        idx_b = tf.keras.Input(shape=(None,), name='idx_b')
        X_a = tf.gather(X_util, idx_a, axis=0)
        X_b = tf.gather(X_util, idx_b, axis=0)
        out = Subtract()([X_b, X_a])
        out = Activation('sigmoid')(out)

        # Create the new model with the additional layers
        m = tf.keras.Model(inputs=[X_input, idx_a, idx_b], outputs=out, name="PairwiseModel")

        # Return the original model as m_infer and the new model as m
        m_infer = model
        return m, m_infer


    # model = createPairwiseModel(config)
    pre_model = setup_model(config)
    #model, model_infer = combine_model(pre_model)
    model = pre_model

    model.compile(optimizer=Adam(config['learning_rate']),
                  loss=BinaryCrossentropy(from_logits=True),
                  metrics=[BinaryAccuracy(threshold=.5)])


    ################################################################################
    # Fit model
    ################################################################################
    hs = model.fit(data_loader, epochs=config['epochs'], verbose=1)
    ################################################################################
    # Evaluate model
    ################################################################################
    logger.info("Testing model")
    loss, acc = model.evaluate(data_loader_test.load(), steps=data_loader_test.steps_per_epoch)
    logger.info(f"Done. Test loss: {loss} - Test Accuracy: {acc}")


    logger.info("--- %s seconds ---" % (time.time() - start_time))
    ###############################################################################
    df = pd.DataFrame({'loss': hs.history['loss'], 'binary_accuracy': hs.history['binary_accuracy']})
    df.to_csv(config['folder_path'] + '/loss_acc.csv', index=False)

