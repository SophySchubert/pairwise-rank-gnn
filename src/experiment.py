import sys
import shutil
import time
import shutil
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
    shutil.copy('src/models/prgnn.py', config['folder_path']+'/model.py')
    ######################################################################
    shutil.copy(f"./src/models/{config['model']}.py", config['folder_path']+ '/nn_model.py')
    ###############################################################################

    # Load data and split it in train and test sets
    train_graphs, test_graphs = get_data(config)
    loader_tr = MyDisjointLoader(train_graphs, batch_size=config['batch_size'], epochs=config['epochs'], seed=config['seed'])
    loader_te = MyDisjointLoader(test_graphs, batch_size=config['batch_size'], epochs=1, seed=config['seed'])

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
    @tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
    def train_step(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(target, predictions) + sum(model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        accuracy = accuracy_fn(target, predictions)
        return loss, accuracy


    epoch = step = loss = accuracy = 0
    epoch_loss = []
    epoch_acc = []
    for batch in loader_tr:
        step += 1
        batch_loss, batch_accuracy = train_step(*batch)
        loss += batch_loss
        accuracy += batch_accuracy
        if step == loader_tr.steps_per_epoch:
            step = 0
            epoch += 1
            epoch_loss.append(np.array(loss / loader_tr.steps_per_epoch))
            epoch_acc.append(np.array(accuracy / loader_tr.steps_per_epoch))
            logger.info(f"Epoch:{epoch}, Loss: {epoch_loss[-1]}, Accuracy: {epoch_acc[-1]}")
            loss = accuracy = 0

    ################################################################################
    # Evaluate model
    ################################################################################
    logger.info("Testing model")
    loss = accuracy = 0
    for batch in loader_te:
        inputs, target = batch
        predictions = model(inputs, training=False)
        loss += loss_fn(target, predictions)
        accuracy += accuracy_fn(target, predictions)
    loss /= loader_te.steps_per_epoch
    accuracy /= loader_te.steps_per_epoch

    logger.info(f"Done. Test loss: {loss}, Test accuracy: {accuracy}, kendalltau: {kendalltau(x=target, y=predictions)}")


    logger.info("--- %s seconds ---" % (time.time() - start_time))
    ###############################################################################
    df = pd.DataFrame({'loss': epoch_loss, 'accuracy': epoch_acc})
    df.to_csv(config['folder_path'] + '/loss_acc.csv', index=False)
    ###############################################################################
    fig, ax1 = plt.subplots()

    # Plot loss
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(range(1, len(epoch_loss) + 1), epoch_loss, color='tab:red', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(range(1, len(epoch_acc) + 1), epoch_acc, color='tab:blue', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Add a title and show the plot
    fig.suptitle('Training Loss and Accuracy per Epoch')
    fig.tight_layout()
    plt.savefig(config['folder_path'] + '/loss_acc.png')
    plt.show()

