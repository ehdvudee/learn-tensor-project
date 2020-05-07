import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

tf.set_random_seed(777)

xTrain = [1,2,3]
yTrain = [1,2,3]

W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

hypothesis = xTrain * W + b

cost = tf.reduce_mean(tf.square(hypothesis - yTrain))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])

        if step % 20 == 0:
            print(step, '\t', cost_val, W_val, b_val)



