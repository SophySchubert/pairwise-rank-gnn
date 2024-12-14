import tensorflow as tf
from spektral.layers import GeneralConv, GlobalSumPool, GraphMasking, ECCConv
from tensorflow import keras

class PRGNN(tf.keras.Model):
    def __init__(self, config):
        super(PRGNN, self).__init__()
        self.graph_masking = GraphMasking()
        # self.conv1 = ECCConv(32, activation="relu")
        # self.conv2 = ECCConv(32, activation="relu")
        # self.conv1 = GeneralConv(32, activation="relu")
        # self.conv2 = GeneralConv(32, activation="relu")
        # self.dense = keras.layers.Dense(1, activation="relu")

        # self.compile(
        #     optimizer=keras.optimizers.Adam(config['learning_rate']),
        #     loss=keras.losses.BinaryCrossentropy(from_logits=True),
        #     metrics=[keras.metrics.BinaryAccuracy(threshold=.5)]
        # )

    def call(self, inputs):
        graph, pref_a, pref_b = inputs
        x, a, e = graph[0], graph[1], graph[2]
        x = self.graph_masking(x)
        print(x.shape)
        print(x)
        # x = self.conv1([x, a, e])
        # x = self.conv2([x, a, e])
        # x_util = self.dense(x)
        # X_a, X_b = self.pref_lookup(x_util, pref_a, pref_b)

        # return X_b - X_a
        return x
    def pref_lookup(self, X, pref_a, pref_b):
        X_a = tf.gather(X, pref_a, axis=0)
        X_b = tf.gather(X, pref_b, axis=0)
        return X_a, X_b