import tensorflow as tf
from spektral.layers import GeneralConv, GlobalSumPool, GraphMasking, ECCConv
from tensorflow import keras

class PRGNN(tf.keras.Model):
    def __init__(self, config):
        super(PRGNN, self).__init__()
        self.masking = GraphMasking()
        self.conv1 = ECCConv(32, activation="relu")
        self.conv2 = ECCConv(32, activation="relu")
        self.pool = GlobalSumPool()
        self.dense = keras.layers.Dense(1, activation="relu")

        self.compile(
            optimizer=keras.optimizers.Adam(config['learning_rate']),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy(threshold=.5)]
        )

    def call(self, inputs):
        graph = inputs[0]
        x, a, e = graph[0], graph[1], graph[2]
        x = self.masking(x)
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        output = self.pool(x)
        output = self.dense(output)
        X_a, X_b = self.pref_lookup(output, inputs[1], input[2])

        return X_a - X_b

    def pref_lookup(self, X, pref_a, pref_b):
        X_a = tf.gather(X, pref_a, axis=0)
        X_b = tf.gather(X, pref_b, axis=0)
        return X_a, X_b