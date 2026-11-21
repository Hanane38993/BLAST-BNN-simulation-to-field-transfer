# MTG-26-0109 revised reproducibility package

## Manuscript
**Domain-Shift and Field-Transfer Assessment of PPV Models Trained on a Numerical Blast Database**

This package contains the revised analysis prepared in response to the Major Revision decision for *Mining Technology: Transactions of the Institutions of Mining and Metallurgy* (manuscript MTG-26-0109).

The original Zenodo v1.0.0 archive remains the record of the originally submitted analysis. This revision package preserves the same raw numerical database, external field observations, and frozen reference predictions, while implementing the reviewer-requested methodological and robustness analyses.

## Principal revision changes

1. **Training-only joint-domain support threshold**
   - The Q-D scaler and nearest-neighbour reference set are fitted on the 1,080-case numerical training subset.
   - The primary support threshold is now the 95th percentile of leave-one-out five-nearest-neighbour distances within the training subset.
   - The held-out numerical test set is reserved for final internal evaluation and no longer defines the support threshold.
   - Revised primary threshold: **0.1493720380**.
   - Revised field classification: **9 supported / 37 weakly supported or sparse**.

2. **Reviewer 3.2 robustness analyses**
   - Limestone-only distance-error correlations.
   - Relative absolute error and absolute log-error analyses.
   - Descriptive study-adjusted regressions.
   - Error-versus-observed-PPV assessment.

3. **Reviewer 4.4 descriptive uncertainty**
   - Within-study stratified bootstrap intervals for the main field metrics.
   - Support-stratified RMSE uncertainty.
   - Exact residual-band coverage counts with Wilson intervals.

4. **Residual-band terminology**
   - The uncertainty-transfer diagnostic is reported as a **validation-derived absolute-residual band**, not as a model-derived Bayesian prediction interval.

5. **Terminology and figures**
   - “Weakly supported/sparse” is used for the density-based support classification.
   - Regenerated main and supplementary figures use the revised classification and non-causal wording.

## Key revised results

- Fixed split: **1,080 training / 231 validation / 232 test**.
- External field observations: **46**.
- Primary Q-D support threshold: **0.1493720380**.
- Supported field observations: **9**.
- Weakly supported/sparse observations: **37**.
- QD-BNN supported RMSE: **17.03 mm/s**.
- QD-BNN weakly supported/sparse RMSE: **166.98 mm/s**.
- Weak-support/supported RMSE ratio: **9.81**.
- Numerical-test residual-band coverage: **219/232 = 94.4%**.
- Field residual-band coverage: **25/46 = 54.3%**.

## Main files

- `BLAST_BNN_Paper2_FINAL_REVISED.py` — complete revised reproducibility script.
- `blast_dataset_FULL.csv` — 1,543-case numerical database.
- `field_external_46.csv` — 46 external field observations.
- `field_predictions_paper.csv` — frozen field predictions used for exact manuscript reproduction.
- `numerical_test_predictions_paper.csv` — frozen numerical-test predictions used for exact manuscript reproduction.
- `requirements.txt` — Python dependencies.
- `LICENSE` — software/data license supplied with the original archive.
- `outputs/` — revised summary files, main/supplementary tables, and PNG/PDF figures.

## Exact revised-manuscript reproduction

Recommended Python: 3.10 or 3.11.

```bash
python BLAST_BNN_Paper2_FINAL_REVISED.py \
  --numerical blast_dataset_FULL.csv \
  --field field_external_46.csv \
  --reference-field field_predictions_paper.csv \
  --reference-test numerical_test_predictions_paper.csv \
  --output revised_outputs \
  --bnn-mode reference \
  --bootstrap-reps 5000 \
  --dpi 800
```

`--bnn-mode reference` uses the frozen predictions and is the mode used for the revised manuscript statistics. The `train` mode retrains the QD-BNN and may show small numerical variation across hardware/software environments.

## Data-use separation

No external field PPV observation is used for model training, preprocessing, hyperparameter selection, model selection, domain-threshold definition, or numerical-validation residual-band calibration. Field observations remain an external transfer-evaluation cohort.

## Public archive

Original archive DOI: https://doi.org/10.5281/zenodo.22917484

Development repository: https://github.com/Hanane38993/BLAST-BNN-simulation-to-field-transfer

The original v1.0.0 archive should remain preserved. After journal resubmission, this revised package can be archived as a new version if desired.
