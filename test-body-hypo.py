import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

tf.disable_v2_behavior()

tf.set_random_seed(777)

xy = np.loadtxt('./datas/ret_body_data_2.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0: 5]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK

print(x_data, "\nx_data shpae: ", x_data.shape)
print(y_data, "\ny_data shape: ", y_data.shape)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 5], name="input")
Y = tf.placeholder(tf.float32, shape=[None, 1], name="output")
W = tf.Variable(tf.random_normal([5, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X, W) + b
tf.identity(hypothesis, "hypothesis")

# Simplified cost/loss function
# Hypothesis
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000015)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initilize global variables in the graph.
sess.run(tf.global_variables_initializer())

lossList = []
epochList = []
for step in range(50000):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    lossList.append(cost_val)
    epochList.append(step)

    if step % 10 == 0:
        print("test : ", _)
        print(step, "cost: ", cost_val, "\nPrediction: \n", hy_val)

saver = tf.train.Saver()
save_path = saver.save(sess, "./ret_model2/saved.ckpt")

# Ask my score
print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[183, 98, 102, 61, 27.5]]}), "\n")

plt.plot(epochList, lossList, 'r-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig("./statistics/body_learan_cost.png")
plt.show()
