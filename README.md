# AIE4011 Recommender System Term Project

> This repository contains some source codes for AIE4011 Recommender System term project, from Sogang University.

## Used Dataset

- [KuaiRec](https://kuairec.com/)
  - `big_matrix.csv` is used for training model.
    - Used 9:1 split to use train/valid dataset. 
  - `small_matrix.csv` is used for evaluating model.

## Environment Info

- Ubuntu 20.04.6 LTS
- Python 3.9.13
- CUDA 11.6

## Python Packages Info

```shell
IPython==8.16.1
ipykernel==6.26.0
jupyter_client==8.5.0
jupyter_core==4.12.0
matplotlib==3.5.2
numpy==1.23.3
pandas==1.4.4
pytorch==1.12.1
scikit-learn==1.1.2
tqdm==4.64.1
```

## Used Model

- Generalized Matrix Factorization (GMF)
- Neural Collaborative Filtering (MLP)
- Neural Matrix Factorization (NeuMF)

## Reference

- [Original author's code](https://github.com/hexiangnan/neural_collaborative_filtering)
- [Referenced code](https://github.com/guoyang9/NCF/tree/master)