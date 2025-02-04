import tensorflow as tf
from tensorflow.keras.layers import Dense
from spektral.layers import ECCConv


class PRGNN(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.conv1 = ECCConv(32, activation="relu")
        self.conv2 = ECCConv(32, activation="relu")
        self.dense = Dense(config['n_out'], activation=None)
        self.subtract = tf.keras.layers.Subtract()
        self.out = tf.keras.layers.Activation('sigmoid')

    def call(self, inputs):#, training=False
        # print(f"inputs:{inputs}")
        # exit(1)
        x, a, e, i, idx_a, idx_b = inputs

        x = tf.cast(x, tf.float32)
        # a = a.with_values(tf.cast(a.values, tf.float32))
        # e = tf.cast(e, tf.float32)
        # print(f"x:{x.shape}")
        # print(f"a:{a.shape}")
        # print(f"e:{e.shape}")
        # print(f"i:{i.shape}")
        # print(f"idx_a:{idx_a.shape}")
        # print(f"idx_b:{idx_b.shape}")

        X = self.conv1([x, a, e])
        X = self.conv2([X, a, e])
        X_util = self.dense(X)
        # print(f"x_util:{X_util}")
        # print(f"len x_util:{len(X_util)}")
        X_a, X_b = self.pref_lookup(X_util, idx_a, idx_b)
        # print(f"pref_lookup output:{X_b - X_a}")
        # out = self.subtract([X_b, X_a])
        # out = self.out(out)
        # out = tf.cast(out > 0.5, tf.float32)
        # print(f"out:{out}")
        return X_b - X_a
        # if training:
        #     return X_b - X_a
        # else:
        #     return X_util

        # return X_b - X_a, X_util

    def pref_lookup(self, X, pref_a, pref_b):
        # print(f"pref_a:{pref_a}")
        # print(f"pref_b:{pref_b}")

        X_a = tf.gather(X, pref_a, axis=0)
        X_b = tf.gather(X, pref_b, axis=0)

        return X_a, X_b