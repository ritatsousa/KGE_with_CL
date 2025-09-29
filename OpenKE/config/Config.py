#coding:utf-8
import random
import numpy as np
from itertools import product
#import tensorflow as tf
import os
import time
import datetime
import ctypes
import json
import tensorflow.compat.v1 as tf
# tf.compat.v1.enable_eager_execution()
tf.disable_v2_behavior()

class Config(object):
	'''
	use ctypes to call C functions from python and set essential parameters.
	'''
	def __init__(self):

		base_file_pos = os.path.abspath(os.path.join(os.path.dirname(__file__), '../release/Base_pos.so'))
		self.lib_pos = ctypes.cdll.LoadLibrary(base_file_pos)
		self.lib_pos.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
		self.lib_pos.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib_pos.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib_pos.testHead.argtypes = [ctypes.c_void_p]
		self.lib_pos.testTail.argtypes = [ctypes.c_void_p]
		self.lib_pos.getTestBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib_pos.getValidBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib_pos.getBestThreshold.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib_pos.test_triple_classification.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.optimizer_pos = None

		base_file_neg = os.path.abspath(os.path.join(os.path.dirname(__file__), '../release/Base_neg.so'))
		self.lib_neg = ctypes.cdll.LoadLibrary(base_file_neg)
		self.lib_neg.sampling.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
		self.lib_neg.getHeadBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib_neg.getTailBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib_neg.testHead.argtypes = [ctypes.c_void_p]
		self.lib_neg.testTail.argtypes = [ctypes.c_void_p]
		self.lib_neg.getTestBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib_neg.getValidBatch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib_neg.getBestThreshold.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.lib_neg.test_triple_classification.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
		self.optimizer_neg = None

		self.epochs_contrastive_learning = 300
		self.test_flag = False
		self.in_path = None
		self.out_path = None
		self.bern = 0
		self.hidden_size = 100
		self.ent_size = self.hidden_size
		self.rel_size = self.hidden_size
		self.train_times = 400
		self.margin = 1.0
		self.nbatches = 100
		self.negative_ent = 1
		self.negative_rel = 0
		self.workThreads = 8
		self.alpha = 0.001
		self.lmbda = 0.000
		self.log_on = 1
		self.exportName = None
		self.importName = None
		self.export_steps = 0
		self.opt_method = "SGD"
		self.test_link_prediction = False
		self.test_triple_classification = False
		self.early_stopping = None # It expects a tuple of the following: (patience, min_delta)

	def init_link_prediction(self):
		r'''
		import essential files and set essential interfaces for link prediction
		'''
		self.lib_pos.importTestFiles()
		self.lib_pos.importTypeFiles()
		self.testTotal_pos = self.lib_pos.getTestTotal()
		self.test_h_pos = np.zeros(self.lib_pos.getEntityTotal(), dtype = np.int64)
		self.test_t_pos = np.zeros(self.lib_pos.getEntityTotal(), dtype = np.int64)
		self.test_r_pos = np.zeros(self.lib_pos.getEntityTotal(), dtype = np.int64)
		self.test_h_addr_pos = self.test_h_pos.__array_interface__['data'][0]
		self.test_t_addr_pos = self.test_t_pos.__array_interface__['data'][0]
		self.test_r_addr_pos = self.test_r_pos.__array_interface__['data'][0]

		self.lib_neg.importTestFiles()
		self.lib_neg.importTypeFiles()
		self.testTotal_neg = self.lib_neg.getTestTotal()
		self.test_h_neg = np.zeros(self.lib_neg.getEntityTotal(), dtype = np.int64)
		self.test_t_neg = np.zeros(self.lib_neg.getEntityTotal(), dtype = np.int64)
		self.test_r_neg = np.zeros(self.lib_neg.getEntityTotal(), dtype = np.int64)
		self.test_h_addr_neg = self.test_h_neg.__array_interface__['data'][0]
		self.test_t_addr_neg = self.test_t_neg.__array_interface__['data'][0]
		self.test_r_addr_neg = self.test_r_neg.__array_interface__['data'][0]


	# prepare for train and test
	def init(self):
		self.trainModel = None

		if self.in_path != None:

			print("Init pos")
			self.lib_pos.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
			self.lib_pos.setBern(self.bern)
			self.lib_pos.setWorkThreads(self.workThreads)
			self.lib_pos.randReset()
			self.lib_pos.importTrainFiles()
			self.relTotal_pos = self.lib_pos.getRelationTotal()
			self.entTotal_pos = self.lib_pos.getEntityTotal()
			self.trainTotal_pos = self.lib_pos.getTrainTotal()
			self.testTotal_pos = self.lib_pos.getTestTotal()
			self.validTotal_pos = self.lib_pos.getValidTotal()
			self.batch_size_pos = int(self.lib_pos.getTrainTotal() / self.nbatches)
			self.batch_seq_size_pos = self.batch_size_pos * (1 + self.negative_ent + self.negative_rel)
			self.batch_h_pos = np.zeros(self.batch_size_pos * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_t_pos = np.zeros(self.batch_size_pos * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_r_pos = np.zeros(self.batch_size_pos * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_y_pos = np.zeros(self.batch_size_pos * (1 + self.negative_ent + self.negative_rel), dtype = np.float32)
			self.batch_h_addr_pos = self.batch_h_pos.__array_interface__['data'][0]
			self.batch_t_addr_pos = self.batch_t_pos.__array_interface__['data'][0]
			self.batch_r_addr_pos = self.batch_r_pos.__array_interface__['data'][0]
			self.batch_y_addr_pos = self.batch_y_pos.__array_interface__['data'][0]

			print("Init neg")
			self.lib_neg.setInPath(ctypes.create_string_buffer(self.in_path.encode(), len(self.in_path) * 2))
			self.lib_neg.setBern(self.bern)
			self.lib_neg.setWorkThreads(self.workThreads)
			self.lib_neg.randReset()
			self.lib_neg.importTrainFiles()
			self.relTotal_neg = self.lib_neg.getRelationTotal()
			self.entTotal_neg = self.lib_neg.getEntityTotal()
			self.trainTotal_neg = self.lib_neg.getTrainTotal()
			self.testTotal_neg = self.lib_neg.getTestTotal()
			self.validTotal_neg = self.lib_neg.getValidTotal()
			self.batch_size_neg = int(self.lib_neg.getTrainTotal() / self.nbatches)
			self.batch_seq_size_neg = self.batch_size_neg * (1 + self.negative_ent + self.negative_rel)
			self.batch_h_neg = np.zeros(self.batch_size_neg * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_t_neg = np.zeros(self.batch_size_neg * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_r_neg = np.zeros(self.batch_size_neg * (1 + self.negative_ent + self.negative_rel), dtype = np.int64)
			self.batch_y_neg = np.zeros(self.batch_size_neg * (1 + self.negative_ent + self.negative_rel), dtype = np.float32)
			self.batch_h_addr_neg = self.batch_h_neg.__array_interface__['data'][0]
			self.batch_t_addr_neg = self.batch_t_neg.__array_interface__['data'][0]
			self.batch_r_addr_neg = self.batch_r_neg.__array_interface__['data'][0]
			self.batch_y_addr_neg = self.batch_y_neg.__array_interface__['data'][0]
		
		if self.test_link_prediction:
			print("Init link prediction")
			self.init_link_prediction()

	def get_ent_total(self):
		return self.entTotal

	def get_rel_total(self):
		return self.relTotal

	def set_lmbda(self, lmbda):
		self.lmbda = lmbda

	def set_optimizer(self, optimizer):
		self.optimizer = optimizer

	def set_opt_method(self, method):
		self.opt_method = method

	def set_test_link_prediction(self, flag):
		self.test_link_prediction = flag

	def set_test_triple_classification(self, flag):
		self.test_triple_classification = flag

	def set_log_on(self, flag):
		self.log_on = flag

	def set_alpha(self, alpha):
		self.alpha = alpha

	def set_in_path(self, path):
		self.in_path = path

	def set_out_files(self, path):
		self.out_path = path

	def set_bern(self, bern):
		self.bern = bern

	def set_dimension(self, dim):
		self.hidden_size = dim
		self.ent_size = dim
		self.rel_size = dim

	def set_ent_dimension(self, dim):
		self.ent_size = dim

	def set_rel_dimension(self, dim):
		self.rel_size = dim

	def set_train_times(self, times):
		self.train_times = times

	def set_nbatches(self, nbatches):
		self.nbatches = nbatches

	def set_margin(self, margin):
		self.margin = margin

	def set_work_threads(self, threads):
		self.workThreads = threads

	def set_ent_neg_rate(self, rate):
		self.negative_ent = rate

	def set_rel_neg_rate(self, rate):
		self.negative_rel = rate

	def set_import_files(self, path):
		self.importName = path

	def set_export_files(self, path_pos, path_neg, steps = 0):
		self.exportName_pos = path_pos
		self.exportName_neg = path_neg
		self.export_steps = steps

	def set_export_steps(self, steps):
		self.export_steps = steps

	def set_early_stopping(self, early_stopping):
		self.early_stopping = early_stopping

	# call C function for sampling
	def sampling(self):
		self.lib_neg.sampling(self.batch_h_addr_neg, self.batch_t_addr_neg, self.batch_r_addr_neg, self.batch_y_addr_neg, self.batch_size_neg, self.negative_ent, self.negative_rel)
		self.lib_pos.sampling(self.batch_h_addr_pos, self.batch_t_addr_pos, self.batch_r_addr_pos, self.batch_y_addr_pos, self.batch_size_pos, self.negative_ent, self.negative_rel)

	# save model
	def save_tensorflow(self):
		with self.graph_pos.as_default():
			with self.sess_pos.as_default():
				self.saver_pos.save(self.sess_pos, self.exportName_pos)
		with self.graph_neg.as_default():
			with self.sess_neg.as_default():
				self.saver_neg.save(self.sess_neg, self.exportName_neg)
	
	def export_variables(self, path = None):
		with self.graph_pos.as_default():
			with self.sess_pos.as_default():
				if path == None:
					self.saver.save(self.sess_pos, self.exportName_pos)
		with self.graph_neg.as_default():
			with self.sess_neg.as_default():
				if path == None:
					self.saver.save(self.sess_neg, self.exportName_neg)		

	def import_variables(self, path_pos, path_neg):
		with self.graph_pos.as_default():
			with self.sess_pos.as_default():
				self.saver_pos.restore(self.sess_pos, path_pos)
		with self.graph_neg.as_default():
			with self.sess_neg.as_default():
				self.saver_neg.restore(self.sess_neg, path_neg)

	def get_parameter_lists(self):
		return self.trainModel_pos.parameter_lists, self.trainModel_neg.parameter_lists

	def get_parameters_by_name_pos(self, var_name):
		with self.graph_pos.as_default():
			with self.sess_pos.as_default():
				if var_name in self.trainModel_pos.parameter_lists:
					return self.sess_pos.run(self.trainModel_pos.parameter_lists[var_name])
				else:
					return None
				
	def get_parameters_by_name_neg(self, var_name):
		with self.graph_neg.as_default():
			with self.sess_neg.as_default():
				if var_name in self.trainModel_neg.parameter_lists:
					return self.sess_neg.run(self.trainModel_neg.parameter_lists[var_name])
				else:
					return None

	def get_parameters(self, mode = "numpy"):
		res = {}
		lists_pos, lists_neg = self.get_parameter_lists()
		for var_name in lists_pos:
			if mode == "numpy":
				res[var_name + "_pos"] = self.get_parameters_by_name_pos(var_name)
			else:
				res[var_name + "_pos"] = self.get_parameters_by_name_pos(var_name).tolist()
		for var_name in lists_neg:
			if mode == "numpy":
				res[var_name + "_neg"] = self.get_parameters_by_name_neg(var_name)
			else:
				res[var_name + "_neg"] = self.get_parameters_by_name_neg(var_name).tolist()	
		return res

	def save_parameters(self, path = None):
		if path == None:
			path = self.out_path
		f = open(path, "w")
		f.write(json.dumps(self.get_parameters("list")))
		f.close()

	def set_parameters_by_name(self, var_name, tensor):
		with self.graph.as_default():
			with self.sess.as_default():
				if var_name in self.trainModel.parameter_lists:
					self.trainModel.parameter_lists[var_name].assign(tensor).eval()

	def set_parameters(self, lists):
		for i in lists:
			self.set_parameters_by_name(i, lists[i])

	def set_model(self, model):
		self.model_pos = model
		self.model_neg = model
		self.graph_pos = tf.Graph()
		self.graph_neg = tf.Graph()
		
		with self.graph_pos.as_default():
			# RITA
			config = tf.ConfigProto(device_count = {'GPU':1})
			self.sess_pos = tf.Session(config=config, graph=self.graph_pos)
			#self.sess = tf.Session()
			with self.sess_pos.as_default():
				#initializer = tf.contrib.layers.xavier_initializer(uniform = True)
				initializer = tf.keras.initializers.glorot_uniform()
				with tf.variable_scope("model", reuse=None, initializer = initializer):
					self.trainModel_pos = self.model_pos(self, self.relTotal_pos, self.entTotal_pos, self.trainTotal_pos, self.testTotal_pos, self.validTotal_pos, self.batch_size_pos, self.batch_seq_size_pos, self.batch_h_pos, self.batch_t_pos, self.batch_r_pos, self.batch_y_pos, self.batch_h_addr_pos, self.batch_t_addr_pos, self.batch_r_addr_pos, self.batch_y_addr_pos)
					if self.optimizer_pos != None:
						pass
					elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
						self.optimizer_pos = tf.train.AdagradOptimizer(learning_rate = self.alpha, initial_accumulator_value=1e-20)
					elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
						self.optimizer_pos = tf.train.AdadeltaOptimizer(self.alpha)
					elif self.opt_method == "Adam" or self.opt_method == "adam":
						self.optimizer_pos = tf.train.AdamOptimizer(self.alpha)
					else:
						self.optimizer_pos = tf.train.GradientDescentOptimizer(self.alpha)
					grads_and_vars = self.optimizer_pos.compute_gradients(self.trainModel_pos.loss)
					self.train_op_pos = self.optimizer_pos.apply_gradients(grads_and_vars)
				self.saver_pos = tf.train.Saver()
				self.sess_pos.run(tf.global_variables_initializer())

		with self.graph_neg.as_default():
			# RITA
			config = tf.ConfigProto(device_count = {'GPU':1})
			self.sess_neg = tf.Session(config=config, graph=self.graph_neg)
			#self.sess = tf.Session()
			with self.sess_neg.as_default():
				#initializer = tf.contrib.layers.xavier_initializer(uniform = True)
				initializer = tf.keras.initializers.glorot_uniform()
				with tf.variable_scope("model", reuse=None, initializer = initializer):
					self.trainModel_neg = self.model_neg(self, self.relTotal_neg, self.entTotal_neg, self.trainTotal_neg, self.testTotal_neg, self.validTotal_neg, self.batch_size_neg, self.batch_seq_size_neg, self.batch_h_neg, self.batch_t_neg, self.batch_r_neg, self.batch_y_neg, self.batch_h_addr_neg, self.batch_t_addr_neg, self.batch_r_addr_neg, self.batch_y_addr_neg)
					if self.optimizer_neg != None:
						pass
					elif self.opt_method == "Adagrad" or self.opt_method == "adagrad":
						self.optimizer_neg = tf.train.AdagradOptimizer(learning_rate = self.alpha, initial_accumulator_value=1e-20)
					elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
						self.optimizer_neg = tf.train.AdadeltaOptimizer(self.alpha)
					elif self.opt_method == "Adam" or self.opt_method == "adam":
						self.optimizer_neg = tf.train.AdamOptimizer(self.alpha)
					else:
						self.optimizer_neg = tf.train.GradientDescentOptimizer(self.alpha)
					grads_and_vars = self.optimizer_neg.compute_gradients(self.trainModel_neg.loss)
					self.train_op_neg = self.optimizer_neg.apply_gradients(grads_and_vars)
				self.saver_neg = tf.train.Saver()
				self.sess_neg.run(tf.global_variables_initializer())


	def train_step(self, batch_h, batch_t, batch_r, batch_y, pos_h, pos_t, pos_r, pos_y, neg_h, neg_t, neg_r, neg_y, type_str):
		if type_str == "pos":
			feed_dict = {
				self.trainModel_pos.batch_h: batch_h,
				self.trainModel_pos.batch_t: batch_t,
				self.trainModel_pos.batch_r: batch_r,
				self.trainModel_pos.batch_y: batch_y, 
				self.trainModel_pos.pos_h: pos_h,
				self.trainModel_pos.pos_t: pos_t,
				self.trainModel_pos.pos_r: pos_r,
				self.trainModel_pos.pos_y: pos_y,
				self.trainModel_pos.neg_h: neg_h,
				self.trainModel_pos.neg_t: neg_t,
				self.trainModel_pos.neg_r: neg_r, 
				self.trainModel_pos.neg_y: neg_y
			}
			
			_, loss = self.sess_pos.run([self.train_op_pos, self.trainModel_pos.loss], feed_dict)

		elif type_str == "neg":
			feed_dict = {
				self.trainModel_neg.batch_h: batch_h,
				self.trainModel_neg.batch_t: batch_t,
				self.trainModel_neg.batch_r: batch_r,
				self.trainModel_neg.batch_y: batch_y,
				self.trainModel_neg.pos_h: pos_h,
				self.trainModel_neg.pos_t: pos_t,
				self.trainModel_neg.pos_r: pos_r,
				self.trainModel_neg.pos_y: pos_y,
				self.trainModel_neg.neg_h: neg_h,
				self.trainModel_neg.neg_t: neg_t,
				self.trainModel_neg.neg_r: neg_r,
				self.trainModel_neg.neg_y: neg_y
			}
			_, loss = self.sess_neg.run([self.train_op_neg, self.trainModel_neg.loss], feed_dict)

		return loss


	def test_step(self, test_h, test_t, test_r, type):

		if type == "pos":
			feed_dict = {
				self.trainModel_pos.predict_h: test_h,
				self.trainModel_pos.predict_t: test_t,
				self.trainModel_pos.predict_r: test_r,
			}
			predict_pos = self.sess_pos.run(self.trainModel_pos.predict, feed_dict)
			return predict_pos
		
		elif type == "neg":
			feed_dict = {
				self.trainModel_neg.predict_h: test_h,
				self.trainModel_neg.predict_t: test_t,
				self.trainModel_neg.predict_r: test_r,
			}
			predict_neg = self.sess_neg.run(self.trainModel_neg.predict, feed_dict)
			return predict_neg


	def generate_neg_samples_pos_model(self, all_heads, all_tails, all_relations):

		unique_heads = np.unique(all_heads)
		unique_tails = np.unique(all_tails)  # Also consider unique tails
		perturbed_heads = np.empty(len(all_tails), dtype=unique_heads.dtype)
		perturbed_tails = np.empty(len(all_tails), dtype=unique_tails.dtype)
		perturbed_heads = np.empty(len(all_tails), dtype=unique_heads.dtype)

		perturb_tails = np.random.rand(len(all_tails)) < 0.5

		# Randomly select 100 unique tails for each head-relation pair
		selected_heads_list = [unique_heads for _ in range(len(all_tails))]
		test_t_perturb_heads = np.repeat(all_tails, len(unique_heads))  # Repeat heads for sampled tails
		test_r_perturb_heads = np.repeat(all_relations, len(unique_heads))  # Repeat relations accordingly
		test_h_perturb_heads = np.concatenate(selected_heads_list)  # Flatten to match test_h and test_r shape
		res_perturb_heads = self.test_step(test_h_perturb_heads, test_t_perturb_heads, test_r_perturb_heads, "neg")

		# Randomly select 100 unique tails for each head-relation pair
		selected_tails_list = [unique_tails for _ in range(len(all_heads))]
		test_h_perturb_tails = np.repeat(all_heads, len(unique_tails))  # Repeat heads for sampled tails
		test_r_perturb_tails = np.repeat(all_relations, len(unique_tails))
		# Repeat relations accordingly
		test_t_perturb_tails = np.concatenate(selected_tails_list)  # Flatten to match test_h and test_r shape
		res_perturb_tails = self.test_step(test_h_perturb_tails, test_t_perturb_tails, test_r_perturb_tails, "neg")

		res_perturb_heads = res_perturb_heads.reshape(len(all_heads), len(unique_heads))
		res_perturb_tails = res_perturb_tails.reshape(len(all_tails), len(unique_tails))
		
		best_heads = np.array([selected_heads_list[i][np.argmax(res_perturb_heads[i])] for i in range(len(all_tails))])
		best_tails = np.array([selected_tails_list[i][np.argmax(res_perturb_tails[i])] for i in range(len(all_heads))])

		perturbed_heads[~perturb_tails] = best_heads[~perturb_tails]  # Keep normal perturbation
		perturbed_heads[perturb_tails] = all_heads[perturb_tails]  # Keep original heads if perturbing tails

		perturbed_tails[perturb_tails] = best_tails[perturb_tails]  # Perturb tails
		perturbed_tails[~perturb_tails] = all_tails[~perturb_tails]
		
		negative_h_pos = np.array(perturbed_heads).reshape(self.batch_size_pos, self.negative_ent + self.negative_rel)
		negative_t_pos = np.array(perturbed_tails).reshape(self.batch_size_pos, self.negative_ent + self.negative_rel)
		negative_r_pos = all_relations.reshape(self.batch_size_pos, self.negative_ent + self.negative_rel)

		return negative_h_pos, negative_t_pos, negative_r_pos


	def generate_neg_samples_neg_model(self, all_heads, all_tails, all_relations):

		unique_heads = np.unique(all_heads)
		unique_tails = np.unique(all_tails)  # Also consider unique tails
		perturbed_heads = np.empty(len(all_tails), dtype=unique_heads.dtype)
		perturbed_tails = np.empty(len(all_tails), dtype=unique_tails.dtype)
		perturbed_heads = np.empty(len(all_tails), dtype=unique_heads.dtype)

		perturb_tails = np.random.rand(len(all_tails)) < 0.5
		
		# Randomly select 100 unique tails for each head-relation pair
		selected_heads_list = [unique_heads for _ in range(len(all_tails))]
		test_t_perturb_heads = np.repeat(all_tails, len(unique_heads))  # Repeat heads for sampled tails
		test_r_perturb_heads = np.repeat(all_relations, len(unique_heads))  # Repeat relations accordingly
		test_h_perturb_heads = np.concatenate(selected_heads_list)  # Flatten to match test_h and test_r shape
		res_perturb_heads = self.test_step(test_h_perturb_heads, test_t_perturb_heads, test_r_perturb_heads, "pos")

		# Randomly select 100 unique tails for each head-relation pair
		selected_tails_list = [unique_tails for _ in range(len(all_heads))]
		test_h_perturb_tails = np.repeat(all_heads, len(unique_tails))  # Repeat heads for sampled tails
		test_r_perturb_tails = np.repeat(all_relations, len(unique_tails))
		# Repeat relations accordingly
		test_t_perturb_tails = np.concatenate(selected_tails_list)  # Flatten to match test_h and test_r shape
		res_perturb_tails = self.test_step(test_h_perturb_tails, test_t_perturb_tails, test_r_perturb_tails, "pos")

		res_perturb_heads = res_perturb_heads.reshape(len(all_heads), len(unique_heads))
		res_perturb_tails = res_perturb_tails.reshape(len(all_tails), len(unique_tails))

		best_heads = np.array([selected_heads_list[i][np.argmax(res_perturb_heads[i])] for i in range(len(all_tails))])
		best_tails = np.array([selected_tails_list[i][np.argmax(res_perturb_tails[i])] for i in range(len(all_heads))])

		perturbed_heads[~perturb_tails] = best_heads[~perturb_tails]  # Keep normal perturbation
		perturbed_heads[perturb_tails] = all_heads[perturb_tails]  # Keep original heads if perturbing tails

		perturbed_tails[perturb_tails] = best_tails[perturb_tails]  # Perturb tails
		perturbed_tails[~perturb_tails] = all_tails[~perturb_tails]

		negative_h_neg = np.array(perturbed_heads).reshape(self.batch_size_neg, self.negative_ent + self.negative_rel)
		negative_t_neg = np.array(perturbed_tails).reshape(self.batch_size_neg, self.negative_ent + self.negative_rel)
		negative_r_neg = all_relations.reshape(self.batch_size_neg, self.negative_ent + self.negative_rel)
		return negative_h_neg, negative_t_neg, negative_r_neg


	def run(self):

		print("Approch: Perturb all heads and tails")

		if self.early_stopping is not None:
			patience, min_delta = self.early_stopping
			best_loss_pos = np.finfo('float32').max
			best_loss_neg = np.finfo('float32').max
			wait_steps = 0

		for times in range(self.train_times):
			loss_pos, loss_neg = 0.0, 0.0
			t_init = time.time()
			t_sample = 0
			t_train = 0

			for i in range(self.nbatches):

				t_s = time.time()
				self.sampling()
				t_e = time.time()
				t_sample = t_sample + t_e - t_s

				positive_h_pos = self.batch_h_pos[0:self.batch_size_pos].reshape(self.batch_size_pos, 1)
				positive_t_pos = self.batch_t_pos[0:self.batch_size_pos].reshape(self.batch_size_pos, 1)
				positive_r_pos = self.batch_r_pos[0:self.batch_size_pos].reshape(self.batch_size_pos, 1)
				positive_y_pos = self.batch_y_pos[0:self.batch_size_pos].reshape(self.batch_size_pos, 1)
				
				positive_h_neg = self.batch_h_neg[0:self.batch_size_neg].reshape(self.batch_size_neg, 1)
				positive_t_neg = self.batch_t_neg[0:self.batch_size_neg].reshape(self.batch_size_neg, 1)
				positive_r_neg = self.batch_r_neg[0:self.batch_size_neg].reshape(self.batch_size_neg, 1)
				positive_y_neg = self.batch_y_neg[0:self.batch_size_neg].reshape(self.batch_size_neg, 1)
				
				negative_h_pos = self.batch_h_pos[self.batch_size_pos:self.batch_seq_size_pos].reshape(self.batch_size_pos, self.negative_ent + self.negative_rel)
				negative_t_pos = self.batch_t_pos[self.batch_size_pos:self.batch_seq_size_pos].reshape(self.batch_size_pos, self.negative_ent + self.negative_rel)
				negative_r_pos = self.batch_r_pos[self.batch_size_pos:self.batch_seq_size_pos].reshape(self.batch_size_pos, self.negative_ent + self.negative_rel)
				negative_y_pos = self.batch_y_pos[self.batch_size_pos:self.batch_seq_size_pos].reshape(self.batch_size_pos, self.negative_ent + self.negative_rel)
					
				negative_h_neg = self.batch_h_neg[self.batch_size_neg:self.batch_seq_size_neg].reshape(self.batch_size_neg, self.negative_ent + self.negative_rel)
				negative_t_neg = self.batch_t_neg[self.batch_size_neg:self.batch_seq_size_neg].reshape(self.batch_size_neg, self.negative_ent + self.negative_rel)
				negative_r_neg = self.batch_r_neg[self.batch_size_neg:self.batch_seq_size_neg].reshape(self.batch_size_neg, self.negative_ent + self.negative_rel)
				negative_y_neg = self.batch_y_neg[self.batch_size_neg:self.batch_seq_size_neg].reshape(self.batch_size_neg, self.negative_ent + self.negative_rel)
				
				if times > self.epochs_contrastive_learning: 

					negative_h_pos, negative_t_pos, negative_r_pos = self.generate_neg_samples_pos_model(self.batch_h_pos[self.batch_size_pos:self.batch_seq_size_pos], self.batch_t_pos[self.batch_size_pos:self.batch_seq_size_pos], self.batch_r_pos[self.batch_size_pos:self.batch_seq_size_pos])
					negative_h_neg, negative_t_neg, negative_r_neg = self.generate_neg_samples_neg_model(self.batch_h_neg[self.batch_size_neg:self.batch_seq_size_neg], self.batch_t_neg[self.batch_size_neg:self.batch_seq_size_neg], self.batch_r_neg[self.batch_size_neg:self.batch_seq_size_neg])

				t_s = time.time()
				loss_pos += self.train_step(self.batch_h_pos, self.batch_t_pos, self.batch_r_pos, self.batch_y_pos, positive_h_pos, positive_t_pos, positive_r_pos, positive_y_pos, negative_h_pos, negative_t_pos, negative_r_pos, negative_y_pos, "pos")	
				loss_neg += self.train_step(self.batch_h_neg, self.batch_t_neg, self.batch_r_neg, self.batch_y_neg, positive_h_neg, positive_t_neg, positive_r_neg, positive_y_neg, negative_h_neg, negative_t_neg, negative_r_neg, negative_y_neg, "neg")
				t_e = time.time()
				t_train = t_train + t_e - t_s

				if self.early_stopping is not None:
					if loss_pos + min_delta < best_loss_pos and loss_neg + min_delta < best_loss_neg:
						best_loss_pos = loss_pos
						best_loss_neg = loss_neg		
						wait_steps = 0
					elif wait_steps < patience:
						wait_steps += 1
					else:
						print('Early stopping. Losses have not been improved enough in {} times'.format(patience))
						break

			t_end = time.time()

			if self.log_on:
				print(f'Epoch: {times}, loss_pos: {loss_pos:0.3f}, loss_neg: {loss_neg:0.3f}, time: {t_end - t_init:0.1f}, sample: {t_sample:0.1f}, train: {t_train:0.1f}')

