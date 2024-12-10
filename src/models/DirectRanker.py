import tensorflow as tf
from tensorflow import keras

class DirectRanker(tf.keras.Model):
    def __init__(self, config):
        super(DirectRanker, self).__init__()
        self.flatten = keras.layers.Flatten(input_shape=(28, 28))
        self.dense_layers = [keras.layers.Dense(128, activation="relu") for _ in range(4)]
        self.output_layer = keras.layers.Dense(1, activation="relu")
        self.compile(
            optimizer=keras.optimizers.Adam(config['learning_rate']),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy(threshold=.0)]
        )

    def call(self, inputs, training=False):
        X_input, pref_a, pref_b = inputs
        X = self.flatten(X_input)
        for dense in self.dense_layers:
            X = dense(X)
        X_utils = self.output_layer(X)
        X_a, X_b = self.pref_lookup(X_utils, pref_a, pref_b)
        return X_b - X_a

    @staticmethod
    def pref_lookup(X, pref_a, pref_b):
        X_a = tf.gather(X, pref_a, axis=0)
        X_b = tf.gather(X, pref_b, axis=0)
        return X_a, X_b