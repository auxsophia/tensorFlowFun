from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd
import numpy as np

# Training info output
tf.logging.set_verbosity(tf.logging.INFO)

# Data sets
APP_TRAINING = "/home/user/tensorFlowFun/iris/iris_read.csv"
APP_TEST     = "/home/user/tensorFlowFun/iris/iris_read.csv"

COLUMNS = ["petal1", "petal2", "petal3", "petal4", "class"]
FEATURES = ["petal1", "petal2", "petal3", "petal4"]
LABEL = "class"

def custom_input_func(data_set):
    # Preprocess data here.
    feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
    #print(feature_cols)
    '''feature_cols = {k:
        tf.SparseTensor(indices=[],     # Put index of where 1 will go here.
            values=[1],                 # Value to put at index.
            shape=[numOfAttributes])    # Row x Col of the matrix (tensor).
        for k in FEATURES}'''
    labels = tf.constant(data_set[LABEL].values)
    #print(labels)
    # ...then return 1) a mapping of feature columns to Tensors with
    # the corresponding feature data, and 2) a Tensor containing labels
    return feature_cols, labels

feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]
training_set   = pd.read_csv(APP_TRAINING, skipinitialspace=True,
                           skiprows=1, names=COLUMNS)
two = custom_input_func(training_set)
with sess as tf.Session():
    sess.run(two.data)
    sess.run(two.target)
