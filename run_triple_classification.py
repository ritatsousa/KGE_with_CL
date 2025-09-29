import pandas as pd
import numpy as np
from rdflib import Graph
import tensorflow as tf

import sklearn
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold

from multiprocessing import cpu_count
n_jobs = cpu_count()

def load_embeddings(checkpoint_path, model_embedding):
    """
    Load entity and relation embeddings from a TensorFlow checkpoint.
    Arguments:
    checkpoint_path (str): Path to the TensorFlow checkpoint directory.
    model_embedding (str): Name of the embedding model. One of {"TransE", "DistMult", "ComplEx"}.
    Outputs:
    If model is ComplEx, returns two dictionaries: ({1: ent1_embeddings, 2: ent2_embeddings}, {1: rel1_embeddings, 2: rel2_embeddings})
    Otherwise, returns two arrays: (ent_embeddings, rel_embeddings).
    """
    reader = tf.train.load_checkpoint(checkpoint_path)
    if "ComplEx" in model_embedding:
        return {1:reader.get_tensor('model/ent1_embeddings'), 2:reader.get_tensor('model/ent2_embeddings')}, {1:reader.get_tensor('model/rel1_embeddings'), 2:reader.get_tensor('model/rel2_embeddings')}
    else:
        return reader.get_tensor('model/ent_embeddings'), reader.get_tensor('model/rel_embeddings')



def train_classifier_cv(ML_model, path_output, y, x):
    """
    Train a ML classifier using 5-fold stratified cross-validation.
    Arguments:
    ML_model (str): Type of ML model to use ("DT", "RF", or "MLP").
    path_output (str): File path to write median cross-validated metrics.
    y (array-like): Labels.
    x (array-like): Feature vectors.
    Outputs:
    Writes median cross-validated metrics (accuracy, precision, recall, F1, AUC) to the specified output file.
    """

    x = np.array(x)
    y = np.array(y)

    acc_list, pr_list, re_list, f1_list, auc_list = [], [], [], [], []

    if ML_model== "DT":
        classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
    elif ML_model == "RF":
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    elif ML_model == "MLP":
        classifier = MLPClassifier(random_state=42)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
    for train_index, test_index in skf.split(x, y):
        train_x, test_x = x[train_index], x[test_index]
        train_y, test_y = y[train_index], y[test_index]

        classifier.fit(train_x, train_y)
        y_pred = classifier.predict(test_x)
        y_pred_proba = classifier.predict_proba(test_x)[:, 1]

        acc_list.append(metrics.accuracy_score(test_y, y_pred))
        pr_list.append(metrics.precision_score(test_y, y_pred, average='weighted'))
        re_list.append(metrics.recall_score(test_y, y_pred, average='weighted'))
        f1_list.append(metrics.f1_score(test_y, y_pred, average='weighted'))
        auc_list.append(metrics.roc_auc_score(test_y, y_pred_proba))

    acc_med, pr_med, re_med, f1_med, auc_med = np.median(acc_list), np.median(pr_list), np.median(re_list), np.median(f1_list), np.median(auc_list) 
    
    with open(path_output, "w") as path_metrics:
        for metric, score in zip(["acc", "pr", "re", "f1", "auc"], [acc_med, pr_med, re_med, f1_med, auc_med]):
            path_metrics.write(f"{metric}: {score:.5f}\n")



def processing_embeddings_dictionaries(model_embedding, set_h, set_t, ent_embeddings_pos, ent_embeddings_pos_neg, repre):

    """
    Process and combine head and tail entity embeddings into feature vectors.
    Arguments:
    model_embedding (str): Embedding model type ("ComplEx" or other).
    set_h (list[int]): List of head entity IDs.
    set_t (list[int]): List of tail entity IDs.
    ent_embeddings_pos (dict or np.ndarray): entity embeddings trained on the positive graph.
    ent_embeddings_pos_neg (dict or np.ndarray): entity embeddings after concatenating the positive embedding and the negative embeddings .
    Outputs:
    x_pos (list[np.ndarray]): entity embeddings trained on the positive graph for the test set.
    x_pos_neg (list[np.ndarray]): entity embeddings after concatenating the positive embedding and the negative embeddings for the test set.
    """
    x_pos, x_pos_neg = [], []

    for h, t in zip(set_h, set_t):

        if "ComplEx" in model_embedding:
            emb_h_pos, emb_t_pos = ent_embeddings_pos[1][h], ent_embeddings_pos[1][t]
            emb_h_pos_neg, emb_t_pos_neg = ent_embeddings_pos_neg[1][h], ent_embeddings_pos_neg[1][t]
        else:    
            emb_h_pos, emb_t_pos = ent_embeddings_pos[h], ent_embeddings_pos[t]
            emb_h_pos_neg, emb_t_pos_neg = ent_embeddings_pos_neg[h], ent_embeddings_pos_neg[t]

        if repre == "concat":   
            emb_pair_pos = np.concatenate((emb_h_pos, emb_t_pos), axis=0)
            emb_pair_pos_neg = np.concatenate((emb_h_pos_neg, emb_t_pos_neg), axis=0)
        elif repre == "hada":
            emb_pair_pos = np.multiply(emb_h_pos, emb_t_pos)  
            emb_pair_pos_neg = np.multiply(emb_h_pos_neg, emb_t_pos_neg)

        x_pos.append(emb_pair_pos)
        x_pos_neg.append(emb_pair_pos_neg)
    
    return x_pos, x_pos_neg



def evaluate_bimodel_embeddings(model_embedding, path_triples_test, path_entities, path_relations, checkpoint_path_pos, checkpoint_path_neg, path_output):

    """
    Evaluate embeddings using RF classification with cross-validation.
    Argumentss:
    model_embedding (str): Embedding model type ("ComplEx" or other).
    path_triples_test (str): File path to the test triples file.
    path_entities (str): File path to the entities mapping file.
    path_relations (str): File path to the relations mapping file.
    checkpoint_path_pos (str): Path to the positive model checkpoint.
    checkpoint_path_neg (str): Path to the negative model checkpoint.
    path_output (str): Directory path to save evaluation metrics.
    Outputs:
    Writes two classification metric files (for positive embeddings and concatenated pos-neg embeddings) in the specified output directory.
    """

    dic_entity2id = {}
    with open(path_entities, "r") as entities_file:
        entities_file.readline()
        for line in entities_file:
            ent , id = line.strip().split("	")
            dic_entity2id[ent] = int(id)
    with open(path_relations, "r") as relations_file:
        relations_file.readline()
        for line in relations_file:
            rel , id = line.strip().split("	")
            dic_entity2id[rel] = int(id)

    set_h, set_t, set_y = [], [], []
    with open(path_triples_test, "r") as triples_test:
        for line in triples_test:
            h, r, t, y = line.strip().split("\t")
            set_h.append(dic_entity2id[h])
            set_t.append(dic_entity2id[t])
            set_y.append(int(y))
        
    ent_embeddings_pos, _ = load_embeddings(checkpoint_path_pos, model_embedding)
    ent_embeddings_neg, _ = load_embeddings(checkpoint_path_neg, model_embedding)

    if "ComplEx" in model_embedding:
        ent_embeddings_pos_neg = {1: np.concatenate((ent_embeddings_pos[1], ent_embeddings_neg[1]), axis=1), 2:np.concatenate((ent_embeddings_pos[2], ent_embeddings_neg[2]), axis=1)}
    else:
        ent_embeddings_pos_neg = np.concatenate((ent_embeddings_pos, ent_embeddings_neg), axis=1)
    
    x_pos, x_pos_neg = processing_embeddings_dictionaries(model_embedding, set_h, set_t, ent_embeddings_pos, ent_embeddings_pos_neg, "hada")

    train_classifier_cv("RF", f"{path_output}KGE_pos-hada-RF_Classification_Metrics.txt", set_y, x_pos)
    train_classifier_cv("RF", f"{path_output}KGE_pos_concate_neg-hada-RF_Classification_Metrics.txt", set_y, x_pos_neg)


    

# ########################## Run GO KG ##########################

model_embeddings = ["TransE(400train_50dim_0.001alpha_1negative_ent_100nbatches_SGD_350clepochs)",
                    "DistMult(400train_50dim_0.001alpha_1negative_ent_100nbatches_Adam_350clepochs)",
                    "ComplEx(400train_50dim_0.001alpha_1negative_ent_100nbatches_Adam_350clepochs)",]

for model_embedding in model_embeddings:
    checkpoint_path_pos = "data/go/" + model_embedding + "/pos_model.vec.tf"
    checkpoint_path_neg = "data/go/" + model_embedding + "/neg_model.vec.tf"
    path_triples_test = "data/go/Test_triples.tsv"
    path_output = "data/go/" + model_embedding + "/"
    path_entities = "data/go/entity2id.txt"
    path_relations = "data/go/relation2id.txt"

    evaluate_bimodel_embeddings(model_embedding, path_triples_test, path_entities, path_relations, checkpoint_path_pos, checkpoint_path_neg, path_output)
