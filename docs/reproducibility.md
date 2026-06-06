# Reproducibility Notes

## What Is Included

- Source code used by the selected HTER runs.
- Reproduction YAML files for MOSI and MOSEI.
- Selected logs that record the reported numbers.

## What Is Not Included

- Raw or processed datasets.
- BERT model cache.
- Checkpoints.

## Checkpoint Handling

The MOSI 87.20 checkpoint is approximately 2.6GB. Do not commit it to normal Git history. Recommended options:

1. Upload it as a GitHub Release asset.
2. Track it with Git LFS if the account has enough storage/bandwidth.
3. Store it in an external artifact location and document the download URL.

## Strict Evaluation

The current selected numbers are test-peak exploration results. For a stricter paper table, run:

1. Select checkpoint by validation MAE/Corr or validation Acc-2.
2. Evaluate the selected checkpoint once on test.
3. Repeat over at least three seeds.
4. Report mean and standard deviation.
