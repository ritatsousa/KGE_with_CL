#coding:utf-8
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Model(object):

	def get_config(self):
		return self.config

	# def get_positive_instance(self, in_batch = True):
	# 	if in_batch:
	# 		return [self.positive_h, self.positive_t, self.positive_r]
		# else:
		# 	return [self.batch_h[0:self.config.batch_size], \
		# 	self.batch_t[0:self.config.batch_size], \
		# 	self.batch_r[0:self.config.batch_size]]

	# def get_negative_instance(self, in_batch = True):
	# 	if in_batch:
	# 		return [self.negative_h, self.negative_t, self.negative_r]
		# else:
		# 	return [self.batch_h[self.config.batch_size:self.config.batch_seq_size],\
		# 	self.batch_t[self.config.batch_size:self.config.batch_seq_size],\
		# 	self.batch_r[self.config.batch_size:self.config.batch_seq_size]]

	# def get_positive_labels(self, in_batch = True):
	# 	if in_batch:
	# 		return self.positive_y
	# 	else:
	# 		return self.batch_y[0:self.config.batch_size]

	# def get_negative_labels(self, in_batch = True):
	# 	if in_batch:
	# 		return self.negative_y
	# 	else:
	# 		return self.batch_y[self.config.batch_size:self.config.batch_seq_size]

	def get_all_instance(self, in_batch = False):
		if in_batch:
			return [tf.transpose(tf.reshape(self.batch_h, [1 + self.config.negative_ent + self.config.negative_rel, -1]), [1, 0]),\
			tf.transpose(tf.reshape(self.batch_t, [1 + self.config.negative_ent + self.config.negative_rel, -1]), [1, 0]),\
			tf.transpose(tf.reshape(self.batch_r, [1 + self.config.negative_ent + self.config.negative_rel, -1]), [1, 0])]
		else:
			return [self.batch_h, self.batch_t, self.batch_r]

	def get_all_labels(self, in_batch = False):
		if in_batch:
			return tf.transpose(tf.reshape(self.batch_y, [1 + self.config.negative_ent + self.config.negative_rel, -1]), [1, 0])
		else:
			return self.batch_y

	def get_predict_instance(self):
		return [self.predict_h, self.predict_t, self.predict_r]

	def input_def(self):
		config = self.config

		self.batch_h = tf.placeholder(tf.int64, [self.batch_seq_size])
		self.batch_t = tf.placeholder(tf.int64, [self.batch_seq_size])
		self.batch_r = tf.placeholder(tf.int64, [self.batch_seq_size])
		self.batch_y = tf.placeholder(tf.float32, [self.batch_seq_size])

		self.pos_h = tf.placeholder(tf.int64, shape=[self.batch_size, 1])  
		self.pos_t = tf.placeholder(tf.int64, shape=[self.batch_size, 1])
		self.pos_r = tf.placeholder(tf.int64, shape=[self.batch_size, 1])
		self.pos_y = tf.placeholder(tf.float32, shape=[self.batch_size, 1])

		self.neg_h = tf.placeholder(tf.int64, shape=[self.batch_size, config.negative_ent + config.negative_rel])
		self.neg_t = tf.placeholder(tf.int64, shape=[self.batch_size, config.negative_ent + config.negative_rel])
		self.neg_r = tf.placeholder(tf.int64, shape=[self.batch_size, config.negative_ent + config.negative_rel])
		self.neg_y = tf.placeholder(tf.float32, shape=[self.batch_size, config.negative_ent + config.negative_rel])

		# self.positive_h = tf.transpose(tf.reshape(self.batch_h[0:self.batch_size], [1, -1]), perm = [1, 0])
		# self.positive_t = tf.transpose(tf.reshape(self.batch_t[0:self.batch_size], [1, -1]), perm = [1, 0])
		# self.positive_r = tf.transpose(tf.reshape(self.batch_r[0:self.batch_size], [1, -1]), perm = [1, 0])
		self.positive_y = tf.transpose(tf.reshape(self.batch_y[0:self.batch_size], [1, -1]), perm = [1, 0])
		# self.negative_h = tf.transpose(tf.reshape(self.batch_h[self.batch_size:self.batch_seq_size], [config.negative_ent + config.negative_rel, -1]), perm = [1, 0])
		# self.negative_t = tf.transpose(tf.reshape(self.batch_t[self.batch_size:self.batch_seq_size], [config.negative_ent + config.negative_rel, -1]), perm = [1, 0])
		# self.negative_r = tf.transpose(tf.reshape(self.batch_r[self.batch_size:self.batch_seq_size], [config.negative_ent + config.negative_rel, -1]), perm = [1, 0])
		self.negative_y = tf.transpose(tf.reshape(self.batch_y[self.batch_size:self.batch_seq_size], [config.negative_ent + config.negative_rel, -1]), perm = [1, 0])
		
		self.predict_h = tf.placeholder(tf.int64, [None])
		self.predict_t = tf.placeholder(tf.int64, [None])
		self.predict_r= tf.placeholder(tf.int64, [None])
		self.parameter_lists = []

	def embedding_def(self):
		pass

	def loss_def(self):
		pass

	def predict_def(self):
		pass

	def __init__(self, config, relTotal, entTotal, trainTotal, testTotal, validTotal, batch_size, batch_seq_size, batch_h, batch_t, batch_r, batch_y, batch_h_addr, batch_t_addr, batch_r_addr, batch_y_addr):
		self.config = config
		self.relTotal = relTotal
		self.entTotal = entTotal
		self.trainTotal = trainTotal
		self.testTotal = testTotal
		self.validTotal = validTotal
		self.batch_size = batch_size
		self.batch_seq_size = batch_seq_size
		self.batch_h = batch_h
		self.batch_t = batch_t
		self.batch_r = batch_r
		self.batch_y = batch_y
		self.batch_h_addr = batch_h_addr
		self.batch_t_addr = batch_t_addr
		self.batch_r_addr = batch_r_addr
		self.batch_y_addr = batch_y_addr
		
		with tf.name_scope("input"):
			self.input_def()

		with tf.name_scope("embedding"):
			self.embedding_def()

		with tf.name_scope("loss"):
			self.loss_def()

		with tf.name_scope("predict"):
			self.predict_def()
