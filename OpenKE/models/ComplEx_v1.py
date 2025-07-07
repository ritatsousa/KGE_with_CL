#coding:utf-8
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from .Model import Model

class ComplEx(Model):

	def embedding_def(self):
		config = self.get_config()
		self.ent1_embeddings = tf.get_variable(name = "ent1_embeddings", shape = [self.entTotal, config.hidden_size], initializer = tf.keras.initializers.glorot_uniform())
		self.rel1_embeddings = tf.get_variable(name = "rel1_embeddings", shape = [self.relTotal, config.hidden_size], initializer = tf.keras.initializers.glorot_uniform())
		self.ent2_embeddings = tf.get_variable(name = "ent2_embeddings", shape = [self.entTotal, config.hidden_size], initializer = tf.keras.initializers.glorot_uniform())
		self.rel2_embeddings = tf.get_variable(name = "rel2_embeddings", shape = [self.relTotal, config.hidden_size], initializer = tf.keras.initializers.glorot_uniform())
		self.parameter_lists = {"ent_re_embeddings":self.ent1_embeddings, \
								"ent_im_embeddings":self.ent2_embeddings, \
								"rel_re_embeddings":self.rel1_embeddings, \
								"rel_im_embeddings":self.rel2_embeddings}
	r'''
	ComplEx extends DistMult by introducing complex-valued embeddings so as to better model asymmetric relations. 
	It is proved that HolE is subsumed by ComplEx as a special case.
	'''
	def _calc(self, e1_h, e2_h, e1_t, e2_t, r1, r2):
		return tf.reduce_sum(e1_h * e1_t * r1 + e2_h * e2_t * r1 + e1_h * e2_t * r2 - e2_h * e1_t * r2, -1, keep_dims = False)

	def loss_def(self):
		#Obtaining the initial configuration of the model
		config = self.get_config()
		#To get positive triples and negative triples for training
		#To get labels for the triples, positive triples as 1 and negative triples as -1
		#The shapes of h, t, r, y are (batch_size, 1 + negative_ent + negative_rel)

		# pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = True)
		# neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = True)
		# pos_y = self.get_positive_labels(in_batch = True)
		# neg_y = self.get_negative_labels(in_batch = True)

		p1_h = tf.nn.embedding_lookup(self.ent1_embeddings, self.pos_h)
		p2_h = tf.nn.embedding_lookup(self.ent2_embeddings, self.pos_h)
		p1_t = tf.nn.embedding_lookup(self.ent1_embeddings, self.pos_t)
		p2_t = tf.nn.embedding_lookup(self.ent2_embeddings, self.pos_t)
		p1_r = tf.nn.embedding_lookup(self.rel1_embeddings, self.pos_r)
		p2_r = tf.nn.embedding_lookup(self.rel2_embeddings, self.pos_r)

		n1_h = tf.nn.embedding_lookup(self.ent1_embeddings, self.neg_h)
		n2_h = tf.nn.embedding_lookup(self.ent2_embeddings, self.neg_h)
		n1_t = tf.nn.embedding_lookup(self.ent1_embeddings, self.neg_t)
		n2_t = tf.nn.embedding_lookup(self.ent2_embeddings, self.neg_t)
		n1_r = tf.nn.embedding_lookup(self.rel1_embeddings, self.neg_r)
		n2_r = tf.nn.embedding_lookup(self.rel2_embeddings, self.neg_r)

		_p_score = self._calc(p1_h, p2_h, p1_t, p2_t, p1_r, p2_r)
		_n_score = self._calc(n1_h, n2_h, n1_t, n2_t, n1_r, n2_r)

		loss_func = tf.reduce_mean(tf.nn.softplus(- self.pos_y * _p_score) + tf.nn.softplus(- self.neg_y * _n_score))
		regul_func = tf.reduce_mean(p1_h ** 2 + p1_t ** 2 + p1_r ** 2 + n1_h ** 2 + n1_t ** 2 + n1_r ** 2 + p2_h ** 2 + p2_t ** 2 + p2_r ** 2 + n2_h ** 2 + n2_t ** 2 + n2_r ** 2) 
		self.loss =  loss_func + config.lmbda * regul_func


