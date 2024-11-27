from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Dense
from spektral.layers import GeneralConv, GlobalSumPool, GlobalAvgPool, GlobalMaxPool

class PRGNN(Model):
    '''
    pairwise rankin gnn
    takes 2 graphs as input and computes a binary output, which determines if the first graph is preferred over the second or otherwise
    '''

    def __init__(self, hidden=64, config=None):
        super().__init__()
        self.input1 = GeneralConv(hidden, activation="relu")
        self.conv1 = GeneralConv(hidden // 2, activation="relu")
        self.input2 = GeneralConv(hidden, activation="relu")
        self.conv2 = GeneralConv(hidden // 2, activation="relu")
        self.pool_sum = GlobalSumPool()
        self.pool_avg = GlobalAvgPool()
        self.pool_max = GlobalMaxPool()
        self.concat = Concatenate()
        self.dense = Dense(hidden // 2 // 2, activation="relu")
        self.output_layer = Dense(1, activation="sigmoid")

    def call(self, inputs):
        graph1, graph2 = inputs
        out1 = self.input1(graph1)
        out1 = self.conv1(out1)
        out1 = self.pool_avg(out1)


        out2 = self.input2(graph2)
        out2 = self.conv2(out2)
        out2 = self.pool_avg(out2)

        out = self.concat([out1, out2])
        out = self.dense(out)
        out = self.output_layer(out)
        return out

