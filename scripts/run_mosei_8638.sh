#!/usr/bin/env bash
set -euo pipefail

python hter/train.py \
  --config_file configs/reproduce/mosei_8638_seed42.yaml \
  --seed 42 \
  --gpu_id "${1:-0}"
