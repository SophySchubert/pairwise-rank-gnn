import tensorflow as tf
from keras.layers import Dropout
from tensorflow import keras
from tensorflow.keras.layers import Dense
from spektral.layers import ECCConv, GraphMasking, GeneralConv, GCNConv, GCSConv, GlobalAvgPool, MinCutPool, GlobalSumPool
from spektral.layers.convolutional import gcn_conv

class PRGNN(tf.keras.Model):
    def __init__(
            self,
            config,
    ):
        super().__init__()
        self.conv1 = ECCConv(32, activation="relu")
        self.conv2 = ECCConv(32, activation="relu")
        self.global_pool = GlobalSumPool()
        self.util = Dense(config['n_out'])
        self.pref_a = keras.Input(shape=(), dtype=tf.int32)
        self.pref_b = keras.Input(shape=(), dtype=tf.int32)

    def call(self, inputs):
        x, a, e, i, idx_a, idx_b = inputs
        x = tf.cast(x, tf.float32)
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        x = self.global_pool([x, i])
        x_util = self.util(x)
        X_a, X_b = self.pref_lookup(x_util, idx_a, idx_b)

        return X_b - X_a


    def pref_lookup(self, X, pref_a, pref_b):

        X_a = tf.gather(X, pref_a, axis=0)
        X_b = tf.gather(X, pref_b, axis=0)

        return X_a, X_b