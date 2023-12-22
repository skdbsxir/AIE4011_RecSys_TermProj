#!/usr/bin/env bash

for SEED in 42 62 365
do
    python3 main.py --seed "$SEED" --name mymodel --model_type GMF --epoch 50
    python3 main.py --seed "$SEED" --name mymodel --model_type NCF --epoch 50
    python3 main.py --seed "$SEED" --name mymodel --model_type NeuMF --epoch 50

    echo "seed $SEED finished"
    sleep 10s
done