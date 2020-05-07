import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = node1 + node2

print("step1")

print(node1)
print(node2)
print(node3)

print("step2")

sess = tf.Session()

print(sess.run([node1, node2]))
print(sess.run(node3))

print("step3")

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

print(sess.run(node3, feed_dict={node1: 3, node2: 4.5}))
print(sess.run(node3, feed_dict={a: [1, 3], b: [2, 4]}))

sess.close()

