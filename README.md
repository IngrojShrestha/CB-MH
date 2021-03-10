## CB-MH

This repository contains code for CB-MH model, model explanation.

|Script|Description|
| ------------- | ------------- |
| CB_MH.py | multi class multi-label text classifier (CNN + BiLSTM + [MultiHeadAttention](https://arxiv.org/abs/1706.03762)) |
| attribution.py | compute the attribution score using [Integrated Gradient approach](https://arxiv.org/abs/1703.01365)|
| sem_filter.py | extract the UMLS concept from given clinical notes and filter the semantic type related to mental illnesses |
