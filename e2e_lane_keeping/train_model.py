import tensorflow as tf
import numpy as np
from tensorflow.core.protobuf import saver_pb2
import cnn_model
import data_handler

import os

class ModelTrainer:
    def __init__(self, epochs = 30, val_split=0.2, L2_norm_const = 0.001, batch_size=100, logs_path='./logs', model_save_path='./save'):
        self.epochs = epochs
        self.val_split = val_split
        self.L2_norm_const = L2_norm_const
        self.batch_size = batch_size

        self.logs_path = logs_path
        self.model_save_path = model_save_path


if __name__ == '__main__':

    logs_path = './logs'
    model_save_path = './save'

    val_split = 0.8
    L2_norm_const = 0.001

    data_handler = data_handler.DataHandler('data/driving_log.csv')

    train_data, val_data = data_handler.get_data_splits(val_split)

    sess = tf.InteractiveSession()

    train_vars = tf.trainable_variables()

    loss = tf.reduce_mean(tf.square(tf.subtract(cnn_model.y_in, cnn_model.y))) + tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * L2_norm_const
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    sess.run(tf.global_variables_initializer())

    tf.summary.scalar("loss", loss)
    merged_summary_op = tf.summary.merge_all()
    saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter('./logs', graph=tf.get_default_graph())

    epochs = 1 #30
    batch_size = 100

    for epoch in range(epochs):
        print("Epoch " + str(epoch))

        for iteration in range(int(len(train_data)/batch_size)):
            print("Iteration " + str(iteration))

            train_data_batch_x, train_data_batch_y = data_handler.get_data_batch(train_data, batch_size, iteration)

            train_step.run(feed_dict={cnn_model.x: train_data_batch_x, cnn_model.y_in: np.expand_dims(train_data_batch_y, axis=1), cnn_model.keep_prob: 0.8})

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
            checkpoint_path = os.path.join(model_save_path, "model.ckpt")
            filename = saver.save(sess, checkpoint_path)
            print("Model saved in file: %s" % filename)