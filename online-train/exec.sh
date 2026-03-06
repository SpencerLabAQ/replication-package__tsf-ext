#!/bin/bash

mkdir -p logs

python -u rnn_exp_online_train.py 2>&1 | tee logs/rnn_exp_online_train.log