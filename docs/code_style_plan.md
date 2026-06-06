# Code Style Plan

The public repository should look like an HTER project rather than an experiment dump, while preserving the numerical behavior of the current best runs.

## Refactor Rules

1. Do not change tensor operations in files used by the selected configs unless a new experiment is planned.
2. Prefer aliases, docstrings, and wrapper modules over destructive renaming.
3. Keep compatibility with existing configs and logs.
4. Keep attribution clear: the code uses a dual-path global-local backbone and adds HTER evidence calibration/routing modules.

## Current Stable Names

| Current File | Public Meaning |
|---|---|
| `GlobalFusionClassifier.py` | Semantic/global evidence path |
| `LocalFusionClassifier.py` | Local/behavioral evidence path |
| `Gate_fusion.py` | HTER evidence calibration and routing |
| `temporal_encoder.py` | A/V temporal residual encoder |

## Planned Non-Breaking Cleanup

- Add a wrapper module `hter/models/hter_model.py` that exposes clear public names:
  - `HTERModel`
  - `SemanticEvidencePath`
  - `BehavioralEvidencePath`
- Keep old file names available for old configs and scripts.
- Add concise docstrings explaining SAE, LBE, and HDE-RER.
- Avoid moving checkpoint-sensitive training code until the paper results are finalized.
