from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Training info output
tf.logging.set_verbosity(tf.logging.INFO)

# Image example: https://www.tensorflow.org/versions/r0.11/images/iris_three_species.jpg

# Data sets
IRIS_TRAINING = "/home/user/tensorFlowCode/iris/iris_training.csv" #"iris_training.csv"
IRIS_TEST = "/home/user/tensorFlowCode/iris/iris_test.csv"         #"iris_test.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TRAINING,
                                                       target_dtype=np.int, features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=IRIS_TEST,
                                                   target_dtype=np.int, features_dtype=np.float32)

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

# By default, the ValidationMonitor below will log both loss and accuracy.
# Will run the model against the test data every 50 steps (default is 100).
# It will use a checkpoint version of the model, so we must have checkpoints.
validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    test_set.data,
    test_set.target,
    every_n_steps=50)

# Build 3 layer DNN with 10, 20, 10 units respectively.
# The config argument adds checkpoints for validation.
# It creates a checkpoint every second because the data set is so small.
# NOTE: The checkpoint is shared between runs!
# If you run the script again and want to start anew, you have to delete the old checkpoint!
# Otherwise you might get OVERFITTING! We can see this as with enough training,
# the predicted output is [1,1] when it should be [1,2]!
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/iris_model",
                                            config=tf.contrib.learn.RunConfig(
                                                save_checkpoints_secs=1))

# Fit model.
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=2000,
               monitors=[validation_monitor]) # monitors takes a list of all montiors to run during training.

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new flower samples.
new_samples = np.array(
    [[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = classifier.predict(new_samples)
print('Predictions: {}'.format(str(y)))
