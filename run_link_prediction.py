import pandas as pd
import numpy as np
from rdflib import Graph
import tensorflow as tf

from multiprocessing import cpu_count
n_jobs = cpu_count()

def get_score_triple(model_embedding, rel_embeddings, ent_embeddings, test_h, test_t, test_r):
    """
    Computes scores for triples (head, relation, tail) under different KGE models. 
    Arguments:
    model_embedding (str): The name of the embedding model (e.g., "TransE", "DistMult", "ComplEx") to determine the score.
    rel_embeddings (dict or ndarray): A dictionary for ComplEx or ndarray for others.
    ent_embeddings (dict or ndarray): A dictionary for ComplEx or ndarray for others.
    test_h (np.ndarray): Array of head entity indices.
    test_t (np.ndarray): Array of tail entity indices.
    test_r (np.ndarray): Array of relation indices.
    Outputs:    
    scores (tf.Tensor): scores for each triple; lower scores imply higher plausibility.
    """

    if "ComplEx" in model_embedding:
        e1_h = tf.gather(ent_embeddings[1], test_h)
        e1_t = tf.gather(ent_embeddings[1], test_t)
        r1 = tf.gather(rel_embeddings[1], test_r)
        e2_h = tf.gather(ent_embeddings[2], test_h)
        e2_t = tf.gather(ent_embeddings[2], test_t)
        r2 = tf.gather(rel_embeddings[2], test_r)
    else:
        h = tf.gather(ent_embeddings, test_h)
        t = tf.gather(ent_embeddings, test_t)
        r = tf.gather(rel_embeddings, test_r)
    
    if "TransE" in model_embedding:
        scores = tf.linalg.norm(h + r - t, axis=-1, ord=2)
    elif "DistMult" in model_embedding:
        scores = -tf.reduce_sum(h * r * t, axis=-1, keepdims = False)
    elif "ComplEx" in model_embedding:
        scores = -tf.reduce_sum(e1_h * e1_t * r1 + e2_h * e2_t * r1 + e1_h * e2_t * r2 - e2_h * e1_t * r2, axis=-1, keepdims = False)

    return(scores)



def test_head(con, h, entityTotal, l_rank, l_reci_rank, l100_tot, l10_tot, l3_tot, l1_tot):
    """
    Updates link prediction metrics by ranking head predictions.
    Arguments:
    con (tf.Tensor): Scores for candidate triples.
    h (int): True head entity.
    entityTotal (list or np.ndarray): List of entity indices considered.
    l_rank (float): counter for MR for head prediction.
    l_reci_rank (float): counter for MRR for head prediction.
    l100_tot (int): Counters for Hits@100 for head prediction.
    l10_tot (int): Counters for Hits@10 for head prediction.
    l3_tot (int): Counters for Hits@3 for head prediction.
    l1_tot (int): Counters for Hits@1 for head prediction.
    Outputs:
    Updated MR, MRR, and Hits metrics as a tuple.
    """
    
    index_h = tf.squeeze(tf.where(tf.equal(entityTotal, h)))
    minimal = tf.gather(con, index_h)
    comparison_values = tf.gather(con, entityTotal)
    
    l_s = tf.reduce_sum(tf.cast(tf.less(comparison_values, minimal), tf.int32), axis=-1)
   
    if l_s < 100:
        l100_tot += 1
    if l_s < 10:
        l10_tot += 1
    if l_s < 3:
        l3_tot += 1
    if l_s < 1:
        l1_tot += 1
    
    l_rank += l_s + 1
    l_reci_rank += 1 / (l_s + 1)

    return l_rank, l_reci_rank, l100_tot, l10_tot, l3_tot, l1_tot
    


def test_tail(con, t, entityTotal, r_rank, r_reci_rank, r100_tot, r10_tot, r3_tot, r1_tot):
    """
    Updates link prediction metrics by ranking tail predictions.
    Arguments:
    con (tf.Tensor): Scores for candidate triples.
    t (int): True tail entity.
    entityTotal (list or np.ndarray): List of entity indices considered.
    r_rank (float): counter for MR for tail prediction.
    r_reci_rank (float): counter for MRR for tail prediction.
    r100_tot (int): Counters for Hits@100 for tail prediction.
    r10_tot (int): Counters for Hits@10 for tail prediction.
    r3_tot (int): Counters for Hits@3 for tail prediction.
    r1_tot (int): Counters for Hits@1 for tail prediction.
    Outputs:
    Updated MR, MRR, and Hits metrics as a tuple.
    """
    index_t = tf.squeeze(tf.where(tf.equal(entityTotal, t)))
    minimal = tf.gather(con, index_t)
    comparison_values = tf.gather(con, entityTotal)
    
    r_s = tf.reduce_sum(tf.cast(tf.less(comparison_values, minimal), tf.int32), axis=-1)

    if r_s < 100:
        r100_tot += 1
    if r_s < 10:
        r10_tot += 1
    if r_s < 3:
        r3_tot += 1
    if r_s < 1:
        r1_tot += 1
    
    r_rank += r_s + 1
    r_reci_rank += 1 / (r_s + 1)    
    return r_rank, r_reci_rank, r100_tot, r10_tot, r3_tot, r1_tot



def test_link_prediction(testTotal, l_rank, l_reci_rank, l100_tot, l10_tot, l3_tot, l1_tot, r_rank, r_reci_rank, r100_tot, r10_tot, r3_tot, r1_tot):
    """
    Finalizes average link prediction metrics after processing all test triples.
    Arguments:
    testTotal (int): Total number of test triples.
    l_rank, l_reci_rank, l100_tot, l10_tot, l3_tot, l1_tot (float): Accumulated metrics for head prediction.
    r_rank, r_reci_rank, r100_tot, r10_tot, r3_tot, r1_tot (float): Accumulated metrics for tail prediction.
    Outputs:
    Final averaged metrics including MR, MRR, Hits@100, Hits@10, Hits@3, Hits@1 for heads, tails, and the average between heads and tails.
    """
    
    l_rank /= testTotal
    r_rank /= testTotal

    l_reci_rank /= testTotal
    r_reci_rank /= testTotal
    
    l100_tot /= testTotal
    l10_tot /= testTotal
    l3_tot /= testTotal
    l1_tot /= testTotal
    
    r100_tot /= testTotal
    r10_tot /= testTotal
    r3_tot /= testTotal
    r1_tot /= testTotal

    mr = (l_rank + r_rank)/2
    mrr = (l_reci_rank + r_reci_rank)/2
    hits_100 = (l100_tot + r100_tot)/2
    hits_10 = (l10_tot + r10_tot)/2
    hits_3 = (l3_tot + r3_tot)/2
    hits_1 = (l1_tot + r1_tot)/2

    return l_rank, l_reci_rank, l100_tot, l10_tot, l3_tot, l1_tot, r_rank, r_reci_rank, r100_tot, r10_tot, r3_tot, r1_tot, mr, mrr, hits_100, hits_10, hits_3, hits_1



def load_embeddings(checkpoint_path, model_embedding):
    """
    Loads entity and relation embeddings from a TensorFlow checkpoint.
    Arguments:
    checkpoint_path (str): Path to the checkpoint directory.
    model_embedding (str): Name of the embedding model (e.g., "TransE", "DistMult", "ComplEx"). 
    Outputs:
    Entity and relation embeddings (dictionary for ComplEx or ndarrays for others).
    """

    reader = tf.train.load_checkpoint(checkpoint_path)
    if "ComplEx" in model_embedding:
        return {1:reader.get_tensor('model/ent1_embeddings'), 2:reader.get_tensor('model/ent2_embeddings')}, {1:reader.get_tensor('model/rel1_embeddings'), 2:reader.get_tensor('model/rel2_embeddings')}
    else:
        return reader.get_tensor('model/ent_embeddings'), reader.get_tensor('model/rel_embeddings')



def evaluate_bimodel_embeddings(model_embedding, path_output_test, path_output_train_pos, path_entities, checkpoint_path_pos, checkpoint_path_neg, path_output):
    """
    Evaluates link prediction metrics on positive and concatenated positive-negative embeddings.
    Arguments:
    model_embedding (str): The name of the embedding model (e.g., "TransE", "DistMult", "ComplEx").
    path_output_test (str): Path to the test triples file.
    path_output_train_pos (str): Path to the positive training triples file.
    path_entities (str): Path to the entities file.
    checkpoint_path_pos (str): Path to the checkpoint with positive embeddings.
    checkpoint_path_neg (str): Path to the checkpoint with negative embeddings.
    path_output (str): Path to save the link prediction metrics.
    Outputs:
    Writes metrics to a CSV file at `path_output`.
    """

    testdata = np.loadtxt(path_output_test, dtype=int, skiprows=1)
    test_h, test_t, test_r = testdata[:, 0], testdata[:, 1], testdata[:, 2]

    traindata_pos = np.loadtxt(path_output_train_pos, dtype=int, skiprows=1)
    train_h_pos, train_t_pos, train_r_pos = traindata_pos[:, 0], traindata_pos[:, 1], traindata_pos[:, 2]

    dic_tails, dic_heads = {}, {}
    for i, (h, r, t) in enumerate(zip(train_h_pos, train_t_pos, train_r_pos)):
        if (h,r) not in dic_tails:
            dic_tails[(h,r)] = [t]
        else:
            dic_tails[(h,r)] = dic_tails[(h,r)] + [t]
        if (t,r) not in dic_heads:
            dic_heads[(t,r)] = [h]
        else:
            dic_heads[(t,r)] = dic_heads[(t,r)] + [h]

    with open(path_entities, "r") as entities_file:
        entities_file.readline()
        entities = [int(line.strip().split("	")[1]) for line in entities_file]
    
    ent_embeddings_pos, rel_embeddings_pos = load_embeddings(checkpoint_path_pos, model_embedding)
    ent_embeddings_neg, rel_embeddings_neg = load_embeddings(checkpoint_path_neg, model_embedding)

    if "ComplEx" in model_embedding:
        ent_embeddings_pos_neg = {1: np.concatenate((ent_embeddings_pos[1], ent_embeddings_neg[1]), axis=1), 2:np.concatenate((ent_embeddings_pos[2], ent_embeddings_neg[2]), axis=1)}
        rel_embeddings_pos_neg = {1: np.concatenate((rel_embeddings_pos[1], rel_embeddings_neg[1]), axis=1), 2:np.concatenate((rel_embeddings_pos[2], rel_embeddings_neg[2]), axis=1)}
    else:
        ent_embeddings_pos_neg = np.concatenate((ent_embeddings_pos, ent_embeddings_neg), axis=1)
        rel_embeddings_pos_neg = np.concatenate((rel_embeddings_pos, rel_embeddings_neg), axis=1)

    l_rank_pos, l_reci_rank_pos, l100_tot_pos, l10_tot_pos, l3_tot_pos, l1_tot_pos = 0, 0, 0, 0, 0, 0
    r_rank_pos, r_reci_rank_pos, r100_tot_pos, r10_tot_pos, r3_tot_pos, r1_tot_pos = 0, 0, 0, 0, 0, 0
    
    l_rank_pos_concat_neg, l_reci_rank_pos_concat_neg, l100_tot_pos_concat_neg, l10_tot_pos_concat_neg, l3_tot_pos_concat_neg, l1_tot_pos_concat_neg = 0, 0, 0, 0, 0, 0
    r_rank_pos_concat_neg, r_reci_rank_pos_concat_neg, r100_tot_pos_concat_neg, r10_tot_pos_concat_neg, r3_tot_pos_concat_neg, r1_tot_pos_concat_neg = 0, 0, 0, 0, 0, 0

    for i, (h, r, t) in enumerate(zip(test_h, test_r, test_t)):

        if (h,r) not in dic_tails:
            dic_tails[(h,r)] = []          
        filtered_entities_t = [ent for ent in entities if ent not in dic_tails[(h,r)]]

        array_h = np.array([h] * len(filtered_entities_t))
        array_r = np.array([r] * len(filtered_entities_t))
        array_t = np.array(filtered_entities_t)
        
        scores_tails_pos = get_score_triple(model_embedding, rel_embeddings_pos, ent_embeddings_pos, array_h, array_t, array_r)
        scores_tails_pos_concat_neg = get_score_triple(model_embedding, rel_embeddings_pos_neg, ent_embeddings_pos_neg, array_h, array_t, array_r)

        r_rank_pos, r_reci_rank_pos, r100_tot_pos, r10_tot_pos, r3_tot_pos, r1_tot_pos = test_tail(scores_tails_pos, t, filtered_entities_t, r_rank_pos, r_reci_rank_pos, r100_tot_pos, r10_tot_pos, r3_tot_pos, r1_tot_pos)
        r_rank_pos_concat_neg, r_reci_rank_pos_concat_neg, r100_tot_pos_concat_neg, r10_tot_pos_concat_neg, r3_tot_pos_concat_neg, r1_tot_pos_concat_neg = test_tail(scores_tails_pos_concat_neg, t, filtered_entities_t, r_rank_pos_concat_neg, r_reci_rank_pos_concat_neg, r100_tot_pos_concat_neg, r10_tot_pos_concat_neg, r3_tot_pos_concat_neg, r1_tot_pos_concat_neg)
        
        #################################

        if (t,r) not in dic_heads:
            dic_heads[(t,r)] = []  
        filtered_entities_h = [ent for ent in entities if ent not in dic_heads[(t,r)]]

        array_t = np.array([t] * len(filtered_entities_h))
        array_r = np.array([r] * len(filtered_entities_h))
        array_h = np.array(filtered_entities_h)
        
        scores_heads_pos = get_score_triple(model_embedding, rel_embeddings_pos, ent_embeddings_pos, array_h, array_t, array_r)
        scores_heads_pos_concat_neg = get_score_triple(model_embedding, rel_embeddings_pos_neg, ent_embeddings_pos_neg, array_h, array_t, array_r)
        
        l_rank_pos, l_reci_rank_pos, l100_tot_pos, l10_tot_pos, l3_tot_pos, l1_tot_pos = test_head(scores_heads_pos, h, filtered_entities_h, l_rank_pos, l_reci_rank_pos, l100_tot_pos, l10_tot_pos, l3_tot_pos, l1_tot_pos)
        l_rank_pos_concat_neg, l_reci_rank_pos_concat_neg, l100_tot_pos_concat_neg, l10_tot_pos_concat_neg, l3_tot_pos_concat_neg, l1_tot_pos_concat_neg = test_head(scores_heads_pos_concat_neg, h, filtered_entities_h, l_rank_pos_concat_neg, l_reci_rank_pos_concat_neg, l100_tot_pos_concat_neg, l10_tot_pos_concat_neg, l3_tot_pos_concat_neg, l1_tot_pos_concat_neg )
        
        if i % 100 == 0:
            print(str(i)+"/"+str(len(test_h)))

    results_pos = test_link_prediction(len(test_h), l_rank_pos, l_reci_rank_pos, l100_tot_pos, l10_tot_pos, l3_tot_pos, l1_tot_pos, r_rank_pos, r_reci_rank_pos, r100_tot_pos, r10_tot_pos, r3_tot_pos, r1_tot_pos)
    results_pos_concat_neg = test_link_prediction(len(test_h), l_rank_pos_concat_neg, l_reci_rank_pos_concat_neg, l100_tot_pos_concat_neg, l10_tot_pos_concat_neg, l3_tot_pos_concat_neg, l1_tot_pos_concat_neg, r_rank_pos_concat_neg, r_reci_rank_pos_concat_neg, r100_tot_pos_concat_neg, r10_tot_pos_concat_neg, r3_tot_pos_concat_neg, r1_tot_pos_concat_neg)
    results = {"pos":results_pos, "pos-concat-neg":results_pos_concat_neg}

    metrics = ["MR(tail)", "MRR(tail)", "Hits@100(tail)", "Hits@10(tail)", "Hits@3(tail)", "Hits@1(tail)",
                "MR(head)", "MRR(head)", "Hits@100(head)", "Hits@10(head)", "Hits@3(head)", "Hits@1(head)",
                "MR", "MRR", "Hits@100", "Hits@10", "Hits@3", "Hits@1"]
    with open(f"{path_output}LP_Metrics.csv", "a") as path_metrics:
        for key in results:
            for metric, value in zip(metrics, results[key]):
                path_metrics.write(f"{key}\t{metric}\t{value}\n")




# ########################## Run Wkidata KG ##########################

model_embeddings = ["TransE(400train_50dim_0.001alpha_1negative_ent_100nbatches_SGD_350clepochs)",
                    "DistMult(400train_50dim_0.001alpha_1negative_ent_100nbatches_Adam_350clepochs)",
                    "ComplEx(400train_50dim_0.001alpha_1negative_ent_100nbatches_Adam_350clepochs)",]

for model_embedding in model_embeddings:
    checkpoint_path_pos = "data/wikidata/" + model_embedding + "/pos_model.vec.tf"
    checkpoint_path_neg = "data/wikidata/" + model_embedding + "/neg_model.vec.tf"
    path_output_test = "data/wikidata/test2id_pos.txt"
    path_output_train_pos = "data/wikidata/train2id_pos.txt"
    path_output = "data/wikidata/" + model_embedding + "/"
    path_entities = "data/wikidata/entity2id.txt"
    evaluate_bimodel_embeddings(model_embedding, path_output_test, path_output_train_pos, path_entities, checkpoint_path_pos, checkpoint_path_neg, path_output)



