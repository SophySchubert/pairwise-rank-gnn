#import tensorflow as tf
#from tensorflow import keras
#import spektral
from tensorflow.keras.model import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GCNConv, GlobalSumPool
from spektral.datasets import TUDataset
from spektral.data import BatchLoader
import networkx as nx
import numpy as np
from datetime import datetime

def now():
    return datetime.now()


class MyFirstGNN(Model):

    def __init__(self, n_hidden, n_labels):
        super().__init__()
        self.graph_conv = GCNConv(n_hidden)
        self.pool = GlobalSumPool()
        self.dropout = Dropout(0.5)
        self.dense = Dense(n_labels, 'softmax')

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.dropout(out)
        out = self.pool(out)
        out = self.dense(out)

        return out


if __name__ == "__main__":
    print(f"Start time:{now()}")

    dataset = TUDataset('PROTEINS')
    loader = BatchLoader(dataset_train, batch_size=32)

    model = MyFirstGNN(32, dataset.n_labels)
    model.compile('adam', 'categorical_crossentropy')


