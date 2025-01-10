import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from spektral.layers import GlobalSumPool, ECCConv, GraphMasking


class PRGNN(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.masking = GraphMasking()
        self.conv1 = ECCConv(32, activation="relu")
        self.conv2 = ECCConv(32, activation="relu")
        self.conv3 = ECCConv(32, activation="relu")
        self.conv4 = ECCConv(32, activation="relu")
        self.global_pool = GlobalSumPool()
        self.dense = Dense(config['n_out'], activation='relu')
        self.pref_a = keras.Input(shape=(), dtype=tf.int32)
        self.pref_b = keras.Input(shape=(), dtype=tf.int32)

        # self.compile(
        #     optimizer=keras.optimizers.Adam(config['learning_rate']),
        #     loss=keras.losses.BinaryCrossentropy(from_logits=True),
        #     metrics=[keras.metrics.BinaryAccuracy(threshold=.5)]
        # )

    def call(self, inputs):
        x, a, e, i, idx_a, idx_b = inputs
        x = tf.cast(x, tf.float32)
        a = tf.cast(a, tf.float32)
        e = tf.cast(e, tf.float32)

        x = self.masking(x)
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        # x = self.conv3([x, a, e])
        # x = self.conv4([x, a, e])

        x = tf.cast(x, tf.int32)
        x = self.global_pool([x, i])
        x = self.dense(x)
        X_a, X_b = self.pref_lookup(x, idx_a, idx_b)

        return X_b - X_a

    def pref_lookup(self, X, pref_a, pref_b):

        X_a = tf.gather(X, pref_a, axis=0)
        X_b = tf.gather(X, pref_b, axis=0)

        return X_a, X_b