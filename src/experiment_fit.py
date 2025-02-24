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
    loader_tr = MyDisjointLoader(train_graphs, batch_size=config['batch_size'], epochs=config['epochs'], seed=config['seed'])
    loader_te = MyDisjointLoader(test_graphs, batch_size=config['batch_size'], epochs=1, seed=config['seed'])
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

    # pairs, targets = sample_preference_pairs(train_graphs)

    ## hat den [Op:GatherV2] Fehler
    # data_loader = CustomDataLoader(train_graphs, pairs, targets, batch_size=32, seed=42)

    ######## setup D############
    # hat den [Op:GatherV2] Fehler

    pairs_and_targets_train = sample_preference_pairs2(train_graphs)
    print(f"pairs_and_targets:{pairs_and_targets_train}")
    data_loader_train = CustomDisjointedLoader(train_graphs, pairs_and_targets_train, config, node_level=False, batch_size=config['batch_size'], epochs=config['epochs'], shuffle=True)

    pairs_and_targets_test = sample_preference_pairs2(test_graphs)
    data_loader_test = CustomDisjointedLoader(test_graphs, pairs_and_targets_test, config, node_level=False, batch_size=config['batch_size'], epochs=1, shuffle=True)

    #########
    # model #
    #########
    def pref_lookup(X, pref_a, pref_b):
        X_a = tf.gather(X, pref_a, axis=0)
        X_b = tf.gather(X, pref_b, axis=0)

        return X_a, X_b

    def combine_model(config):
        from tensorflow.keras.layers import Input, Dense, Subtract, Activation
        from spektral.layers import ECCConv

        x_in = Input(shape=(None,9))
        a_in = Input(shape=(None,None))
        e_in = Input(shape=(None,3))
        i_in = Input(shape=(None,1))#can be ignored
        idx_a = Input(shape=(None,), dtype=tf.int32)
        idx_b = Input(shape=(None,), dtype=tf.int32)

        # x_in = tf.cast(x_in, tf.float32)
        # a_in = tf.cast(a_in, tf.float32) #a_in.with_values(tf.cast(a_in.values, tf.float32))
        # e_in = tf.cast(e_in, tf.float32)

        outs = ECCConv(32, activation='relu')([x_in, a_in, e_in])
        outs = ECCConv(32, activation='relu')([outs, a_in, e_in])
        X_util = Dense(config['n_out'], activation=None)(outs)

        X_a, X_b = pref_lookup(X_util, idx_a, idx_b)
        out = X_b - X_a

        # Create the new model with the additional layers
        # m_infer = tf.keras.Model(inputs=[x_in, a_in, e_in, i_in], outputs=X_util, name="InferenceModel")
        m = tf.keras.Model(inputs=[x_in, a_in, e_in, i_in, idx_a, idx_b], outputs=out, name="PairwiseModel")

        return m#, m_infer


    #model= combine_model(config) #, model_infer
    model = setup_model(config)

    model.compile(optimizer=Adam(config['learning_rate']),
                  loss=BinaryCrossentropy(from_logits=True),
                  metrics=[BinaryAccuracy(threshold=.5)])


    ################################################################################
    # Fit model
    ################################################################################
    hs = model.fit(data_loader_train.load(), epochs=config['epochs'], verbose=1)
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
    ###############################################################################
    # import not matplotlib.pyplot as plt
    # fig, ax1 = plt.subplots()
    #
    # # Plot loss
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Loss', color='tab:red')
    # ax1.plot(range(1, len(epoch_loss) + 1), epoch_loss, color='tab:red', label='Loss')
    # ax1.tick_params(axis='y', labelcolor='tab:red')
    #
    # # Create a second y-axis for accuracy
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Accuracy', color='tab:blue')
    # ax2.plot(range(1, len(epoch_acc) + 1), epoch_acc, color='tab:blue', label='Accuracy')
    # ax2.tick_params(axis='y', labelcolor='tab:blue')
    #
    # # Add a title and show the plot
    # fig.suptitle('Training Loss and Accuracy per Epoch')
    # fig.tight_layout()
    # plt.savefig(config['folder_path'] + '/loss_acc.png')
    # plt.show()

