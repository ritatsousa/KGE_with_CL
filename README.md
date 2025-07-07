# Improving Knowledge Graph Embeddings through Contrastive Learning with Negative Statements

This repository provides an implementation of a method for improving knowledge graph embeddings (KGEs) by incorporating contrastive learning with explicit negative statements. It includes KGE training scripts, evaluation for link prediction and protein-protein interaction prediction.

## Pre-requisites
* install python 3.9;
* install python libraries by running the following command:  ```pip install -r req.txt```.

## Methodology

<img src="https://github.com/ritatsousa/KGE_with_CL/blob/main/methodology.png" width="50%"/>  


Train the KGE model with contrastive learning by running:
```
python3 run_kge.py
```

Run the link prediction evaluation by running:
```
python3 run_link_prediction.py
```


Run the triple classification evaluation (particularly, protein-protein interaction prediction) by running:
```
python3 run_triple_classification.py
```
