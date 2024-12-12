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
        graph1, graph2 = inputs[0], inputs[1]
        print(graph1)
        print(graph2)
        exit(1)
        x, a, e = graph1
        x = self.masking(x)
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        output = self.pool(x)
        output = self.dense(output)

        return output