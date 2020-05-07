import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
tf.disable_v2_behavior()

tf.set_random_seed(777)

xy = np.loadtxt('./datas/노원_상계.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 1: -1]
y_data = xy[:, [-1]]

plt.scatter(x_data[[]], x_data[[1]])
plt.show()

# Make sure the shape and data are OK

print(x_data, "\nx_data shpae: ", x_data.shape)
print(y_data, "\ny_data shape: ", y_data.shape)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3], name="input")
Y = tf.placeholder(tf.float32, shape=[None, 1], name="output")
W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
tf.identity(hypothesis, "hypothesis")

# Simplified cost/loss function
# Hypothesis
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000000129)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initilize global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(100000):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 500 == 0:
        print(step, "cost: ", cost_val, "\nPrediction: \n", hy_val)

saver = tf.train.Saver()
save_path = saver.save(sess, "./ret_model/saved.ckpt")

# Ask my score
print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[84.87, 6, 1997]]}), "\n")
print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[53.07, 5, 1988]]}), "\n")
print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[84.7, 10, 2016]]}), "\n")
print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[84.8223, 11, 2006]]}), "\n")
