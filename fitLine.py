import tensorflow as tf
import numpy as np

# Training info output
tf.logging.set_verbosity(tf.logging.INFO)

# For summaries - TensorBoard - directory for summaries
if tf.gfile.Exists('/tmp/fitLine'):
  tf.gfile.DeleteRecursively('/tmp/fitLine')
tf.gfile.MakeDirs('/tmp/fitLine')

# For summaries - TensorBoard
def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

# Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Summary data
variable_summaries(W, 'weights')
variable_summaries(b, 'biases')
variable_summaries(y, 'trainingY')

# Minimize the mean squared errors.
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Summary of loss
tf.scalar_summary('loss', loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.merge_all_summaries()
test_writer = tf.train.SummaryWriter('tmp/fitLine/test')
tf.initialize_all_variables().run()

# Fit the line.
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        summary = sess.run(merged)
        train_writer.add_summary(summary, step)
        print(step, sess.run(W), sess.run(b))

train_writer.close()

# Learns best fit is W: [0.1], b: [0.3]
