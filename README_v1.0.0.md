# BLAST-BNN Simulation-to-Field Transfer

Reproducibility materials for the study:

**Simulation-to-Field Transfer and Domain-Shift Assessment of Numerical-Data-Trained Models for Blast-Induced Peak Particle Velocity**

## Repository contents

This repository contains the code and data required to reproduce the simulation-to-field transfer analysis for blast-induced peak particle velocity (PPV).

The analysis evaluates four PPV prediction approaches:

- Scaled-distance power-law model
- Random Forest
- XGBoost
- Reduced QD-BNN

The models are developed using numerical data only. The external field observations are used only for independent evaluation.

The reduced QD-BNN uses two inputs: charge weight (`Q`) and blast-to-monitoring distance (`D`). Therefore, this study is a simulation-to-field transfer and domain-shift assessment and should not be interpreted as field validation of the original 12-input BLAST-BNN model.

## Main files

- `BLAST_BNN_Paper2_FINAL.py` — complete reproducibility analysis
- `blast_dataset_FULL.csv` — numerical database
- `field_external_46.csv` — external field observations
- `field_predictions_paper.csv` — frozen field predictions used for exact manuscript reproduction
- `numerical_test_predictions_paper.csv` — frozen numerical-test predictions used for exact manuscript reproduction
- `requirements.txt` — Python dependencies

## Primary analysis

The numerical dataset is divided into:

- 1,080 training cases
- 231 validation cases
- 232 test cases

The external field dataset contains 46 observations.

The primary joint Q-D support analysis identifies:

- 8 supported field observations
- 38 sparse/OOD field observations

## Exact manuscript reproduction

Run:

```bash
python BLAST_BNN_Paper2_FINAL.py \
    --numerical blast_dataset_FULL.csv \
    --field field_external_46.csv \
    --reference-field field_predictions_paper.csv \
    --reference-test numerical_test_predictions_paper.csv \
    --output paper2_outputs \
    --bnn-mode reference
````

Reference mode uses the frozen paper predictions for all four models so that the manuscript results are reproduced consistently across software environments.

## Citation 

If you use this repository, please cite:

GOUDJIL, H. (2026). *BLAST-BNN Simulation-to-Field Transfer v1.0.0* (Version v1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22917484

Repository archive:

https://doi.org/10.5281/zenodo.22917484

```
