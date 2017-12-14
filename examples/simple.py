from __future__ import print_function
import tensorflow as tf

# created variables
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

# create the linear model
linear_model = W*x + b

# create a session
sess = tf.Session()

# init variables
init = tf.global_variables_initializer()
sess.run(init)

# evaluate linear model
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

print("first run : ", sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

#change values of x and y
changeW = tf.assign(W, [-1])
changeB = tf.assign(b, [1])

sess.run([changeB, changeW])

print("perfect run : ", sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


