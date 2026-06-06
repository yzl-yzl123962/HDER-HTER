#!/usr/bin/env bash
set -euo pipefail

python hter/train.py \
  --config_file configs/reproduce/mosi_8720_tcn_k5_seed5.yaml \
  --seed 5 \
  --gpu_id "${1:-0}"
