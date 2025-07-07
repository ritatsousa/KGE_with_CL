#coding:utf-8
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .Model import Model

class DistMult(Model):
	r'''
	DistMult is based on the bilinear model where each relation is represented by a diagonal rather than a full matrix. 
	DistMult enjoys the same scalable property as TransE and it achieves superior performance over TransE.
	'''
	def _calc(self, h, t, r):
		return tf.reduce_sum(h * r * t, -1, keep_dims = False)

	def embedding_def(self):
		config = self.get_config()
		self.ent_embeddings = tf.get_variable(name = "ent_embeddings", shape = [self.entTotal, config.hidden_size], initializer = tf.keras.initializers.glorot_uniform())
		self.rel_embeddings = tf.get_variable(name = "rel_embeddings", shape = [self.relTotal, config.hidden_size], initializer = tf.keras.initializers.glorot_uniform())
		self.parameter_lists = {"ent_embeddings":self.ent_embeddings, \
								"rel_embeddings":self.rel_embeddings}
	def loss_def(self):
		config = self.get_config()

		# pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = True)
		# neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = True)
		
		# pos_y = self.get_positive_labels(in_batch = True)
		# neg_y = self.get_negative_labels(in_batch = True)
		
		p_h = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_h)
		p_t = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_t)
		p_r = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_r)
		n_h = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_h)
		n_t = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_t)
		n_r = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_r)

		_p_score = self._calc(p_h, p_t, p_r)
		_n_score = self._calc(n_h, n_t, n_r)

		loss_func = tf.reduce_mean(tf.nn.softplus(- self.pos_y * _p_score) + tf.nn.softplus(- self.neg_y * _n_score))
		
		regul_func = tf.reduce_mean(p_h ** 2 + p_t ** 2 + p_r ** 2 + n_h ** 2 + n_t ** 2 + n_r ** 2) 
		self.loss =  loss_func + config.lmbda * regul_func


