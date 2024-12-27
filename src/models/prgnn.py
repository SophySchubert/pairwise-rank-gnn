import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from spektral.layers import GlobalSumPool, ECCConv


class PRGNN(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.conv1 = ECCConv(32, activation="relu")
        self.conv2 = ECCConv(32, activation="relu")
        self.global_pool = GlobalSumPool()
        self.dense = Dense(config['n_out'], activation='relu')
        self._dense = Dense(128, activation='relu')
        self.pref_a = keras.Input(shape=(), dtype=tf.int32)
        self.pref_b = keras.Input(shape=(), dtype=tf.int32)

        # self.compile(
        #     optimizer=keras.optimizers.Adam(config['learning_rate']),
        #     loss=keras.losses.BinaryCrossentropy(from_logits=True),
        #     metrics=[keras.metrics.BinaryAccuracy(threshold=.5)]
        # )

    def call(self, inputs):
        ### inputs: ([x, a, e, i], idx_a, idx_b)
        print("call")
        # print(f"len(inputs): {len(inputs)}")
        # print(f"inputs: {inputs}")
        # print(f"x: {inputs[0][0]}")
        # print(f"a: {inputs[0][1]}")
        # print(f"e: {inputs[0][2]}")
        # print(f"i: {inputs[0][3]}")
        # print(f"idx_a: {inputs[1]}")
        # print(f"idx_b: {inputs[2]}")
        # print("call")
        x, a, e, i, idx_a, idx_b = inputs
        # print("types +  tensors")
        # print(f"x:{x}")
        # print(f"a:{a}")
        # print(f"e:{e}")
        # print(f"i:{i}")
        # print(f"idx_a:{idx_a}")
        # print(f"idx_b:{idx_b}")

        # x = self.conv1([x, a, e])
        # x = self.conv2([x, a, e])
        # output = self.global_pool([x, i])
        output = self._dense(x)
        output = self.dense(output)
        X_a, X_b = self.pref_lookup(output, idx_a, idx_b)
        # print(X_a)
        # print(X_b)
        # print(X_b - X_a)
        # exit(1)

        return X_b - X_a

    def pref_lookup(self, X, pref_a, pref_b):

        X_a = tf.gather(X, pref_a, axis=0)
        X_b = tf.gather(X, pref_b, axis=0)

        return X_a, X_b