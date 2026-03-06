#!/bin/bash

mkdir -p ../logs

python -u tuning.py 2>&1 | tee ../logs/tuning.log

python -u ts_analysis.py --aggregation daily 2>&1 | tee ../logs/ts_analysis_daily.log

python -u baseline.py --aggregation daily 2>&1 | tee ../logs/baseline_daily.log

python -u stat_exp.py --aggregation daily 2>&1 | tee ../logs/stat_exp_daily.log

python -u rnn_exp.py --aggregation daily 2>&1 | tee ../logs/rnn_exp_daily.log

python -u pretrained.py --aggregation daily 2>&1 | tee ../logs/pretrained_daily.log