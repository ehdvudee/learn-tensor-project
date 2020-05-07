import tensorflow.compat.v1 as tf
#https://lueseypid.tistory.com/248

tf.disable_v2_behavior()
tf.set_random_seed(777)

with tf.Session() as sess:
  saver = tf.train.import_meta_graph("./ret_model2/saved.ckpt.meta")
  saver.restore(sess, tf.train.latest_checkpoint("./ret_model2"))

  graph = tf.get_default_graph()

  X = sess.graph.get_tensor_by_name("input:0")
  Y = sess.graph.get_tensor_by_name("output:0")
  hypothesis = sess.graph.get_tensor_by_name("hypothesis:0")

  print("Your score will be ", sess.run(hypothesis, feed_dict={X: [[174.1, 98, 105, 58, 26.8]]}), "\n")
