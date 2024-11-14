import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader
from spektral.datasets import TUDataset
from spektral.models import GeneralGNN
from spektral.layers import GCNConv, GlobalSumPool

devices = tf.config.list_physical_devices('GPU')
if len(devices) > 0:
    tf.config.experimental.set_memory_growth(devices[0], True)

batch_size = 32
learning_rate = 0.01
epochs = 400

data = TUDataset('PROTEINS')

np.random.shuffle(data)
split = int(0.8 * len(data))
data_train, data_test = data[:split], data[split:]

loader_train = DisjointLoader(data_train, batch_size=batch_size, epochs=epochs)
loader_test = DisjointLoader(data_test, batch_size=batch_size)

model = GeneralGNN(data.n_labels, activation='softmax')
optimizer = Adam(learning_rate=learning_rate)
loss_function = CategoricalCrossentropy()

class MyGNN(Model):
    def __init__(self, n_hidden, n_labels):
        super().__init__()
        self.graph_conv = GCNConv(n_hidden)
        self.pool = GlobalSumPool()
        self.dropout = Dropout(0.5)
        self.dense = Dense(n_labels, 'softmax')

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.dropout(out)
        out = self.pool(out)
        out = self.dense(out)

        return out

model = MyGNN(n_hidden=32, n_labels=data.n_labels)
model.compile('adam', 'categorical_crossentropy')

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
        print(
            "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Test loss: {:.3f} - Test acc: {:.3f}".format(
                epoch, *np.mean(results, 0), *results_test
            )
        )
        results = []

results_test = evaluate(loader_test)
print("Final results - Loss: {:.3f} - Acc: {:.3f}".format(*results_test))