import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from spektral.layers import GlobalSumPool, ECCConv, GraphMasking


class PRGNN(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.conv1 = ECCConv(32, activation="relu")
        self.conv2 = ECCConv(32, activation="relu")
        self.dense = Dense(config['n_out'], activation=None)

    def call(self, inputs):
        x, a, e, i, idx_a, idx_b = inputs

        x = tf.cast(x, tf.float32)
        a = a.with_values(tf.cast(a.values, tf.float32))
        e = tf.cast(e, tf.float32)

        X = self.conv1([x, a, e])
        X = self.conv2([X, a, e])
        X_util = self.dense(X)
        X_a, X_b = self.pref_lookup(X_util, idx_a, idx_b)

        return X_b - X_a#, X_util

    def pref_lookup(self, X, pref_a, pref_b):

        X_a = tf.gather(X, pref_a, axis=0)
        X_b = tf.gather(X, pref_b, axis=0)

        return X_a, X_b