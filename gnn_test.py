import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam

from spektral.data import DisjointLoader
from spektral.datasets import TUDataset
from spektral.models import GeneralGNN

devices = tf.config.list_physical_devices('GPU')
if len(devices) > 0:
    print(devices)
