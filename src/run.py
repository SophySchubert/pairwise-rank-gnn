from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader
from spektral.datasets import TUDataset
from spektral.layers import GCNConv, GlobalSumPool

from misc import setup_logger, now



class MyGNN(Model):
    def __init__(self, n_hidden, n_labels):
        super().__init__()
        self.graph_conv = GCNConv(n_hidden)
        self.pool = GlobalSumPool()
        self.dropout = Dropout(0.5)
        self.dense = Dense(n_labels, activation='softmax')

    def call(self, inputs):
        x, a, i = inputs
        x = self.graph_conv([x, a])
        x = self.pool([x, i])
        x = self.dropout(x)
        x = self.dense(x)
        return x



class TrainingHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('accuracy'))



if __name__ == "__main__":
    logger = setup_logger()
    logger.info(f"Start time: {now()}")

    batch_size = 32
    learning_rate = 0.01
    epochs = 400
    optimizer = Adam(learning_rate=learning_rate)
    loss_function = CategoricalCrossentropy()

    # Load and preprocess the dataset
    dataset = TUDataset('PROTEINS')
    np.random.shuffle(dataset)
    split = int(0.8 * len(dataset))
    data_train, data_test = dataset[:split], dataset[split:]

    loader_train = DisjointLoader(data_train, batch_size=32, epochs=400)
    loader_test = DisjointLoader(data_test, batch_size=32)


    @tf.function(input_signature=loader_train.tf_signature(), experimental_relax_shapes=True)
    def train_step(input, targets):
        with tf.GradientTape() as tape:
            predictions = model(input, training=True)
            loss = loss_function(targets, predictions) + sum(model.losses)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        accuracy = tf.reduce_mean(categorical_accuracy(targets, predictions))

        return loss, accuracy


    def evaluate(loader):
        output = []
        step = 0

        while step < loader.steps_per_epoch:
            step += 1
            inputs, target = loader.__next__()
            prediction = model(inputs, training=False)
            outputs = (
                loss_function(target, prediction),
                tf.reduce_mean(categorical_accuracy(target, prediction)),
                len(target)
            )
            output.append(outputs)

            if step == loader.steps_per_epoch:
                output = np.array(output)
                return np.average(output[:, :-1], 0, weights=output[:, -1])

    # Initialize the model, optimizer, and loss function
    model = MyGNN(n_hidden=32, n_labels=dataset.n_labels)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    epoch = step = 0
    results = []
    for batch in loader_train:
        step += 1
        loss, accuracy = train_step(*batch)
        results.append((loss, accuracy))

        if step == loader_train.steps_per_epoch:
            step = 0
            epoch += 1
            results_test = evaluate(loader_test)
            logger.info(
                "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Test loss: {:.3f} - Test acc: {:.3f}".format(
                    epoch, *np.mean(results, 0), *results_test
                )
            )
            results = []

    results_test = evaluate(loader_test)
    logger.info("Final results - Loss: {:.3f} - Acc: {:.3f}".format(*results_test))