from spektral.layers import ECCConv, GlobalSumPool, GraphMasking
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

class Net(Model):
    def __init__(self, config):
        super().__init__()
        self.n_out = config['n_out']
        self.masking = GraphMasking()
        self.conv1 = ECCConv(32, activation="relu")
        self.conv2 = ECCConv(32, activation="relu")
        self.global_pool = GlobalSumPool()
        self.dense = Dense(self.n_out)

    def call(self, inputs):
        x, a, e = inputs
        x = self.masking(x)
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        output = self.global_pool(x)
        output = self.dense(output)

        return output