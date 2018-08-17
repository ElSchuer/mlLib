import tensorflow as tf
import scipy

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add = a + b

with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs', sess.graph)
    output = sess.run(add, {a: [1, 3], b:[2, 4]})
    writer.close()
    print('Adding a and b:', output)
