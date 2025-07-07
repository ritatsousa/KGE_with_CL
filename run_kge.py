import random
import pandas as pd
import numpy as np
import pickle
import rdflib
import json
import os 
import sys
import importlib
from rdflib import Graph

from OpenKE import models
import OpenKE.config as config_module

import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def ensure_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)



def train_openke_embeddings(model_embedding, path_output, dic_hyperpara):

    """
    Trains a KGE model (TransE, DistMult, or ComplEx) with specified hyperparameters.
    Arguments:
    model_embedding (str): Name of the embedding model to train. One of {"TransE", "DistMult", "ComplEx"}.
    path_output (str): Base directory where output files and models will be stored.
    dic_hyperpara (dict): Dictionary of hyperparameters, expected keys include:
    - 'train_times': int, number of training epochs
    - 'hidden_size': int, embedding dimension
    - 'alpha': float, learning rate
    - 'negative_ent': int, number of negative entities
    - 'n_batches': int, number of batches
    - 'opt_method': str, optimizer (e.g., 'SGD', 'Adam')
    - 'epochs_contrastive_learning': int, epochs for contrastive learning
    - 'lmbda': float, regularization parameter
    Outputs:
    Saves model files, embeddings (embedding.vec.json), checkpoints adn training logs in the `path_output`.
    """

    path_output_model = f"{path_output}{model_embedding}({dic_hyperpara['train_times']}train_{dic_hyperpara['hidden_size']}dim_{dic_hyperpara['alpha']}alpha_{dic_hyperpara['negative_ent']}negative_ent_{dic_hyperpara['n_batches']}nbatches_{dic_hyperpara['opt_method']}_{dic_hyperpara['epochs_contrastive_learning']}clepochs)"
    
    os.makedirs(path_output_model + "/", exist_ok=True)
    with open(path_output_model + "/Parameters.txt", "w") as path_parameters:
        path_parameters.write(str(dic_hyperpara))

    con = config_module.Config()
    con.bern = dic_hyperpara["bern"]
    con.hidden_size = dic_hyperpara["hidden_size"]
    con.ent_size = dic_hyperpara["hidden_size"]
    con.rel_size = dic_hyperpara["hidden_size"]
    con.train_times = dic_hyperpara["train_times"]
    con.nbatches = dic_hyperpara["n_batches"]
    con.negative_ent = dic_hyperpara["negative_ent"]
    con.negative_rel = dic_hyperpara["negative_rel"]
    con.alpha = dic_hyperpara["alpha"]
    con.lmbda = dic_hyperpara["lmbda"]
    con.opt_method = dic_hyperpara["opt_method"]
    con.epochs_contrastive_learning = dic_hyperpara["epochs_contrastive_learning"]
    con.test_link_prediction = False
    con.test_triple_classification = False

    con.set_in_path(path_output)
    con.set_export_files(path_output_model + "/pos_model.vec.tf", path_output_model + "/neg_model.vec.tf", 0)
    con.set_out_files(path_output_model + "/embedding.vec.json")
    
    con.init()
    if model_embedding == "TransE":
        con.set_model(models.TransE)
    elif model_embedding == "DistMult":
        con.set_model(models.DistMult)
    elif model_embedding == "ComplEx":
        con.set_model(models.ComplEx)
    
    log_file = f"{path_output_model}/Training_log.txt"
    with open(log_file, "w") as f:
        sys.stdout = f 
        con.run()
        sys.stdout = sys.__stdout__ 

    con.save_tensorflow()



########################## Run Wkidata KG ##########################

path_output = "data/wikidata/"

model_embeddings = ["TransE"]
list_hyperparameters = [{"bern":0, "hidden_size":50, "train_times":400, "n_batches":100, "negative_ent":1, "negative_rel":0, "alpha": 0.001, "lmbda":0, "opt_method":"SGD", "epochs_contrastive_learning":350}]
for model_embedding in model_embeddings:
    for dic_hyperpara in list_hyperparameters:
        train_openke_embeddings(model_embedding, path_output, dic_hyperpara)

model_embeddings = ["DistMult", "ComplEx"]
list_hyperparameters = [{"bern":0, "hidden_size":50, "train_times":400, "n_batches":100, "negative_ent":1, "negative_rel":0, "alpha": 0.001, "lmbda":0.05, "opt_method":"Adam", "epochs_contrastive_learning":350}]
for model_embedding in model_embeddings:
    for dic_hyperpara in list_hyperparameters:
        train_openke_embeddings(model_embedding, path_output, dic_hyperpara)


########################## Run GO KG ##########################

path_output = "data/go/"

model_embeddings = ["TransE"]
list_hyperparameters = [{"bern":0, "hidden_size":50, "train_times":400, "n_batches":100, "negative_ent":1, "negative_rel":0, "alpha": 0.001, "lmbda":0, "opt_method":"SGD", "epochs_contrastive_learning":350}]
for model_embedding in model_embeddings:
    for dic_hyperpara in list_hyperparameters:
        train_openke_embeddings(model_embedding, path_output, dic_hyperpara)

model_embeddings = ["DistMult", "ComplEx"]
list_hyperparameters = [{"bern":0, "hidden_size":50, "train_times":400, "n_batches":100, "negative_ent":1, "negative_rel":0, "alpha": 0.001, "lmbda":0.05, "opt_method":"Adam", "epochs_contrastive_learning":350}]
for model_embedding in model_embeddings:
    for dic_hyperpara in list_hyperparameters:
        train_openke_embeddings(model_embedding, path_output, dic_hyperpara)
