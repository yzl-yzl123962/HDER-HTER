# Selected Results

These results are copied from the selected logs in `experiments/logs`.

## MOSI

Config: `configs/reproduce/mosi_8720_tcn_k5_seed5.yaml`  
Log: `experiments/logs/mosi_8720_tcn_k5_seed5.log`

```text
Has0 Acc-2: 84.99
Has0 F1:    84.72
Non0 Acc-2: 87.20
Non0 F1:    87.01
Acc-5:      56.27
Acc-7:      48.54
MAE:        0.6948
Corr:       0.8002
```

## MOSEI

Config: `configs/reproduce/mosei_8638_seed42.yaml`  
Log: `experiments/logs/mosei_8638_seed42.log`

```text
Has0 Acc-2: 82.89
Has0 F1:    83.12
Non0 Acc-2: 86.38
Non0 F1:    86.42
Acc-5:      54.99
Acc-7:      53.36
MAE:        0.5369
Corr:       0.7747
```

## Protocol Note

The selected logs report test-peak results across epochs. This is useful for internal comparison with earlier exploratory runs, but strict paper reporting should also include validation-selected checkpoints and mean/std over multiple seeds.
