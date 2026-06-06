# HTER

HTER is a multimodal sentiment analysis codebase for heterogeneous cross-modal disagreement modeling. It builds on a dual-path global-local evidence backbone and adds lightweight evidence calibration and routing modules:

- **SAE**: semantic anchor evidence calibration for global semantic evidence.
- **LBE**: local behavioral evidence preservation for audio/visual temporal cues.
- **HDE-RER**: heterogeneous disagreement estimation with bounded evidence routing.
- **A/V-TCN**: residual temporal refinement for pre-extracted audio and visual features.

The repository contains source code, selected reproduction configs, and selected experiment logs. Datasets and checkpoints are not committed.

## Main Results

The numbers below follow the current internal **test-peak** protocol used during model exploration. For strict generalization reporting, run validation-selected checkpoint selection and multi-seed evaluation.

| Dataset | Setting | Seed | Non0 Acc-2 | F1 | Acc-5 | Acc-7 | MAE | Corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MOSI | HTER + A/V-TCN | 5 | 87.20 | 87.01 | 56.27 | 48.54 | 0.6948 | 0.8002 |
| MOSEI | HTER | 42 | 86.38 | 86.42 | 54.99 | 53.36 | 0.5369 | 0.7747 |

Selected logs are stored in [`experiments/logs`](experiments/logs).

## Repository Layout

```text
hter/
  core/                 dataset, metrics, scheduler, utilities
  models/               SAE/LBE/HDE-RER model implementation
  train.py              standard training entrypoint
  train_nockpt.py       training without checkpoint saving
  train_missing_diag.py missing-modality diagnostic script
configs/reproduce/      selected reproduction configs
experiments/logs/       selected result logs
scripts/                convenience launch and parsing scripts
docs/                   result notes and source manifest
```

Useful notes:

- Paper/story draft: [`docs/paper_blueprint.md`](docs/paper_blueprint.md)
- Result summary: [`docs/results.md`](docs/results.md)
- Reproducibility notes: [`docs/reproducibility.md`](docs/reproducibility.md)
- Code cleanup plan: [`docs/code_style_plan.md`](docs/code_style_plan.md)

## Data

Expected dataset paths in the provided configs:

```text
MOSI:  /root/autodl-tmp/DPDF-LQ/dataset/MOSI/aligned_50.pkl
MOSEI: /root/autodl-tmp/DPDF-LQ/dataset/MOSEI/aligned_50.pkl
```

Adjust `dataset.dataPath` in the YAML files if your data is stored elsewhere.

## Environment

The experiments were run on an RTX 3090 rental machine with PyTorch, Transformers, einops, scikit-learn, and tensorboardX installed. A minimal environment can be created with:

```bash
pip install torch transformers einops scikit-learn numpy pyyaml tqdm tensorboardX
```

## Reproduction Commands

Run from the repository root:

```bash
python hter/train.py --config_file configs/reproduce/mosi_8720_tcn_k5_seed5.yaml --seed 5 --gpu_id 0
python hter/train.py --config_file configs/reproduce/mosei_8638_seed42.yaml --seed 42 --gpu_id 0
```

Convenience shell scripts are also available under [`scripts`](scripts).

## Checkpoint Artifact

The MOSI 87.20 checkpoint is about 2.6GB, so it is not suitable for normal Git history. Use Git LFS, a GitHub Release asset, or an external artifact store.

Original experiment-machine path:

```text
/root/autodl-tmp/hder_best_archive/HDER-MODULAR-mosi-tcn160-31290_20260605_tcn160/best_cpk/Non0_acc_2_0.8720_epoch_21_seed_5.pth
```

## Attribution

HTER uses a reproduced dual-path global-local evidence backbone inspired by DPDF-LQ-style multimodal sentiment fusion. The added components in this repository focus on disagreement-aware evidence calibration and bounded routing over that evidence space.
