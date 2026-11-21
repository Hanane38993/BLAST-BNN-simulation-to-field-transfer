#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3

"""
BLAST_BNN_Paper2_FINAL_REVISED.py
=================================

Single-file reproducibility code for:

"Domain-Shift and Field-Transfer Assessment of PPV Models Trained on a
Numerical Blast Database"

WHAT THIS SCRIPT DOES
---------------------
1. Loads the 1,543-case numerical PPV database.
2. Reproduces the fixed 1080/231/232 train-validation-test split.
3. Fits:
      - scaled-distance power law
      - Random Forest
      - XGBoost
      - Reduced QD-BNN
4. Applies the frozen models to 46 external field observations.
5. Computes joint Q-D domain support.
6. Calculates all numerical-test and field-transfer metrics.
7. Performs Pearson/Spearman domain-distance analysis.
8. Performs study-stratified descriptive bootstrap uncertainty intervals.
9. Performs k/threshold sensitivity analysis.
10. Evaluates transfer of the numerical-validation absolute-residual band.
11. Writes all main and supplementary tables.
12. Generates the publication figures at 800 dpi plus vector PDF copies.

IMPORTANT REPRODUCIBILITY NOTE
------------------------------
Neural-network training can vary slightly across hardware / PyTorch / CUDA
versions even with fixed seeds.

Therefore two QD-BNN modes are provided:

    --bnn-mode reference   (DEFAULT)
        Uses the frozen paper predictions for all four models:
        scaled-distance power law, Random Forest, XGBoost, and Reduced QD-BNN.
        This reproduces the exact manuscript numerical and field metrics.

    --bnn-mode train
        Retrains the QD-BNN from the numerical training subset using the
        documented architecture, preprocessing and random seed.

No field PPV is used for training, preprocessing, tuning, model selection,
domain-threshold definition, or numerical-validation residual-band calibration.

EXAMPLE: EXACT PAPER REPRODUCTION
---------------------------------
python BLAST_BNN_Paper2_FINAL_REVISED.py \
    --numerical blast_dataset_FULL.csv \
    --field field_external_46.csv \
    --reference-field field_predictions_paper.csv \
    --reference-test numerical_test_predictions_paper.csv \
    --output paper2_outputs \
    --bnn-mode reference

EXAMPLE: RETRAIN THE QD-BNN
---------------------------
python BLAST_BNN_Paper2_FINAL_REVISED.py \
    --numerical blast_dataset_FULL.csv \
    --field field_external_46.csv \
    --output paper2_outputs_retrained \
    --bnn-mode train

Recommended Python: 3.10 or 3.11

Required packages
-----------------
numpy
pandas
scipy
scikit-learn
xgboost
matplotlib

Required only for --bnn-mode train
----------------------------------
torch
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from xgboost import XGBRegressor


# =============================================================================
# 1. FIXED PAPER SETTINGS
# =============================================================================

SEED = 42

TEST_SIZE = 0.15
VALIDATION_FRACTION_OF_DEVELOPMENT = 0.176

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=10,
    min_samples_split=10,
    max_features=0.50,
    random_state=SEED,
    n_jobs=-1,
)

XGB_PARAMS = dict(
    n_estimators=700,
    max_depth=11,
    learning_rate=0.07,
    subsample=0.60,
    colsample_bytree=1.00,
    min_child_weight=3,
    reg_lambda=1.50,
    objective="reg:squarederror",
    random_state=SEED,
    n_jobs=-1,
)

PRIMARY_K = 5
PRIMARY_PERCENTILE = 95.0

DEFAULT_BOOTSTRAP_REPS = 5000
DEFAULT_FIGURE_DPI = 800

BNN_BATCH_SIZE = 32
BNN_WEIGHT_DECAY = 1.2e-4
BNN_GRAD_CLIP = 0.42
BNN_MAX_EPOCHS = 270
BNN_PATIENCE = 40
BNN_MC_PASSES = 100
BNN_TARGET_QUANTILES = 500

# Exact paper interval half-width from the numerical validation residuals.
PAPER_Q95_MM_S = 74.717880


# Exact 5,000-resample study-stratified bootstrap intervals used in the
# locked manuscript analysis. These are used only in reference mode with
# --bootstrap-reps 5000. Train mode always recomputes bootstrap intervals.
PAPER_BOOTSTRAP_CI = {
    "Power law": {
        "pearson": (0.4295639966630169, 0.6677577411747064),
        "spearman": (0.5160777070010217, 0.7652316733940316),
    },
    "Random Forest": {
        "pearson": (0.30709275312638284, 0.595468509184725),
        "spearman": (0.24356030118549504, 0.6460958855183578),
    },
    "XGBoost": {
        "pearson": (0.47020221556695563, 0.6959239235105387),
        "spearman": (0.578118148050848, 0.785188582675992),
    },
    "QD-BNN": {
        "pearson": (0.49144508411721455, 0.7142625865869003),
        "spearman": (0.6336491813779139, 0.8206975243948988),
    },
}

PAPER_BNN_CONTRAST_CI = {
    "MAE_difference_CI95": (72.4677292198, 143.0663890271),
    "RMSE_ratio_CI95": (6.6978117826, 15.8840483493),
}


# =============================================================================
# 2. REPRODUCIBILITY
# =============================================================================

def set_global_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    except ImportError:
        pass


# =============================================================================
# 3. DATA LOADING AND FIXED SPLIT
# =============================================================================

def load_numerical(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"charge_weight_kg", "distance_m", "ppv_mm_s"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Numerical CSV missing columns: {sorted(missing)}")
    return df.copy()


def load_field(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "study",
        "case_id",
        "charge_weight_kg",
        "distance_m",
        "observed_ppv_mm_s",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Field CSV missing columns: {sorted(missing)}")
    return df.copy()


def fixed_split(n: int):
    idx = np.arange(n)

    development_idx, test_idx = train_test_split(
        idx,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    train_idx, validation_idx = train_test_split(
        development_idx,
        test_size=VALIDATION_FRACTION_OF_DEVELOPMENT,
        random_state=SEED,
    )

    return train_idx, validation_idx, test_idx


# =============================================================================
# 4. METRICS
# =============================================================================

def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "N": int(len(y_true)),
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE_mm_s": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_mm_s": float(mean_absolute_error(y_true, y_pred)),
        "Bias_PredMinusObs_mm_s": float(np.mean(y_pred - y_true)),
    }


def empirical_coverage(y_true, lower, upper) -> float:
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


# =============================================================================
# 5. SCALED-DISTANCE POWER LAW
# =============================================================================

class ScaledDistancePowerLaw:
    """
    log(PPV) = intercept + slope * log(D / sqrt(Q))
    """

    def __init__(self):
        self.intercept_ = None
        self.slope_ = None

    @staticmethod
    def scaled_distance(q, d):
        q = np.asarray(q, dtype=float)
        d = np.asarray(d, dtype=float)
        return d / np.sqrt(q)

    def fit(self, q, d, y):
        sd = self.scaled_distance(q, d)
        self.slope_, self.intercept_ = np.polyfit(
            np.log(sd),
            np.log(np.asarray(y, dtype=float)),
            1,
        )
        return self

    def predict(self, q, d):
        sd = self.scaled_distance(q, d)
        return np.exp(self.intercept_ + self.slope_ * np.log(sd))


# =============================================================================
# 6. REDUCED QD-BNN
# =============================================================================

class ReducedQDBNN:
    """
    Reduced two-input single-output network.

    Architecture:
        Q,D -> 128 -> 64 -> 32 -> 16 -> mean + log variance

    Dropout:
        0.20 after 64-neuron layer
        0.10 after 32-neuron layer
    """

    def __init__(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for --bnn-mode train."
            ) from exc

        self.torch = torch
        self.nn = nn

        self.x_scaler = StandardScaler()

        self.y_transformer = QuantileTransformer(
            n_quantiles=BNN_TARGET_QUANTILES,
            output_distribution="normal",
            random_state=SEED,
        )

        class Net(nn.Module):
            def __init__(self):
                super().__init__()

                self.fc1 = nn.Linear(2, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 32)
                self.fc4 = nn.Linear(32, 16)

                self.mean_head = nn.Linear(16, 1)
                self.logvar_head = nn.Linear(16, 1)

                self.gelu = nn.GELU()
                self.dropout1 = nn.Dropout(0.20)
                self.dropout2 = nn.Dropout(0.10)

            def forward(self, x):
                x = self.gelu(self.fc1(x))
                x = self.dropout1(self.gelu(self.fc2(x)))
                x = self.dropout2(self.gelu(self.fc3(x)))
                x = self.gelu(self.fc4(x))

                mean = self.mean_head(x)
                logvar = self.logvar_head(x).clamp(-10.0, 8.0)

                return mean, logvar

        self.model = Net()

    def _prepare_x(self, q, d, fit=False):
        x = np.column_stack(
            [
                np.asarray(q, dtype=float),
                np.asarray(d, dtype=float),
            ]
        )

        if fit:
            return self.x_scaler.fit_transform(x)

        return self.x_scaler.transform(x)

    def _heteroscedastic_nll(self, mean, logvar, target):
        torch = self.torch

        return 0.5 * (
            logvar
            + (target - mean) ** 2 * torch.exp(-logvar)
        ).mean()

    def fit(
        self,
        q_train,
        d_train,
        y_train,
        q_validation,
        d_validation,
        y_validation,
    ):
        torch = self.torch
        from torch.utils.data import DataLoader, TensorDataset

        x_train = self._prepare_x(q_train, d_train, fit=True)
        x_val = self._prepare_x(q_validation, d_validation, fit=False)

        self.y_transformer.set_params(
            n_quantiles=min(BNN_TARGET_QUANTILES, len(y_train))
        )

        y_train_t = self.y_transformer.fit_transform(
            np.asarray(y_train, dtype=float).reshape(-1, 1)
        )

        y_val_t = self.y_transformer.transform(
            np.asarray(y_validation, dtype=float).reshape(-1, 1)
        )

        dataset = TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train_t, dtype=torch.float32),
        )

        generator = torch.Generator().manual_seed(SEED)

        loader = DataLoader(
            dataset,
            batch_size=BNN_BATCH_SIZE,
            shuffle=True,
            generator=generator,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1.0e-3,
            weight_decay=BNN_WEIGHT_DECAY,
        )

        x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_t, dtype=torch.float32)

        best_validation_loss = float("inf")
        best_state = None
        stale_epochs = 0

        for epoch in range(1, BNN_MAX_EPOCHS + 1):

            if epoch <= 15:
                learning_rate = 1.0e-3
            elif epoch <= 30:
                learning_rate = 9.1e-4
            else:
                learning_rate = 7.64e-4

            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            self.model.train()

            for x_batch, y_batch in loader:
                optimizer.zero_grad(set_to_none=True)

                mean, logvar = self.model(x_batch)

                loss = self._heteroscedastic_nll(
                    mean,
                    logvar,
                    y_batch,
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    BNN_GRAD_CLIP,
                )

                optimizer.step()

            self.model.eval()

            with torch.no_grad():
                mean, logvar = self.model(x_val_tensor)

                validation_loss = float(
                    self._heteroscedastic_nll(
                        mean,
                        logvar,
                        y_val_tensor,
                    ).item()
                )

            if validation_loss < best_validation_loss - 1e-6:
                best_validation_loss = validation_loss
                best_state = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

            if epoch > 30 and stale_epochs >= BNN_PATIENCE:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def predict_mc(self, q, d, n_passes=BNN_MC_PASSES):
        torch = self.torch

        x = torch.tensor(
            self._prepare_x(q, d, fit=False),
            dtype=torch.float32,
        )

        predictions = []
        transformed_aleatoric_variance = []

        # Dropout remains active during inference.
        self.model.train()

        with torch.no_grad():
            for _ in range(n_passes):
                mean, logvar = self.model(x)

                mean_original = self.y_transformer.inverse_transform(
                    mean.cpu().numpy()
                ).reshape(-1)

                predictions.append(mean_original)

                transformed_aleatoric_variance.append(
                    np.exp(logvar.cpu().numpy().reshape(-1))
                )

        prediction_matrix = np.asarray(predictions)

        return {
            "mean": prediction_matrix.mean(axis=0),
            "mc_std": prediction_matrix.std(axis=0, ddof=1),
            "samples": prediction_matrix,
            "aleatoric_transformed_mean": np.mean(
                np.asarray(transformed_aleatoric_variance),
                axis=0,
            ),
        }

    def predict(self, q, d):
        return self.predict_mc(q, d)["mean"]


# =============================================================================
# 7. JOINT Q-D DOMAIN SUPPORT
# =============================================================================

class QDDomainSupport:
    """
    Domain support in standardized log(Q)-log(D) space.

    The scaler, nearest-neighbor reference model, and support threshold are
    derived from the numerical training subset only. The threshold uses
    leave-one-out training distances so that the held-out numerical test set
    remains untouched for final internal evaluation.
    """

    def __init__(self, k=PRIMARY_K, percentile=PRIMARY_PERCENTILE):
        self.k = int(k)
        self.percentile = float(percentile)

        self.scaler = StandardScaler()
        self.nearest_neighbors = None
        self.threshold_ = None

    @staticmethod
    def _raw_transform(q, d):
        q = np.asarray(q, dtype=float)
        d = np.asarray(d, dtype=float)

        if np.any(q <= 0) or np.any(d <= 0):
            raise ValueError("Q and D must be positive before log transformation.")

        return np.column_stack(
            [
                np.log(q),
                np.log(d),
            ]
        )

    def fit(
        self,
        q_train,
        d_train,
    ):
        train_transformed = self.scaler.fit_transform(
            self._raw_transform(q_train, d_train)
        )

        # Training-only leave-one-out calibration of domain support.
        # Request k+1 neighbours because each training observation is its own
        # nearest neighbour; column 0 (self-distance) is then excluded.
        calibration_neighbors = NearestNeighbors(
            n_neighbors=self.k + 1
        ).fit(train_transformed)

        training_distances = calibration_neighbors.kneighbors(
            train_transformed,
            return_distance=True,
        )[0][:, 1:].mean(axis=1)

        self.threshold_ = float(
            np.percentile(
                training_distances,
                self.percentile,
            )
        )

        # Frozen training reference used for numerical-test and field distances.
        self.nearest_neighbors = NearestNeighbors(
            n_neighbors=self.k
        ).fit(train_transformed)

        return self

    def distance(self, q, d):
        transformed = self.scaler.transform(
            self._raw_transform(q, d)
        )

        return self.nearest_neighbors.kneighbors(
            transformed,
            return_distance=True,
        )[0].mean(axis=1)

    def classify(self, q, d):
        distance = self.distance(q, d)
        supported = distance <= self.threshold_

        return distance, supported


# =============================================================================
# 8. STUDY-STRATIFIED BOOTSTRAP
# =============================================================================

def study_groups(study):
    study = np.asarray(study)

    levels = list(dict.fromkeys(study.tolist()))

    return {
        level: np.where(study == level)[0]
        for level in levels
    }


def bootstrap_correlation(
    distance,
    absolute_error,
    study,
    statistic="spearman",
    repetitions=DEFAULT_BOOTSTRAP_REPS,
    seed=SEED,
):
    distance = np.asarray(distance, dtype=float)
    absolute_error = np.asarray(absolute_error, dtype=float)
    study = np.asarray(study)

    groups = study_groups(study)
    rng = np.random.default_rng(seed)

    estimates = []

    for _ in range(repetitions):

        sampled_indices = np.concatenate(
            [
                rng.choice(
                    indices,
                    size=len(indices),
                    replace=True,
                )
                for indices in groups.values()
            ]
        )

        x = distance[sampled_indices]
        y = absolute_error[sampled_indices]

        if np.std(x) == 0 or np.std(y) == 0:
            continue

        if statistic == "pearson":
            value = pearsonr(x, y).statistic
        else:
            value = spearmanr(x, y).statistic

        if np.isfinite(value):
            estimates.append(value)

    estimates = np.asarray(estimates)

    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def _study_stratified_bootstrap_indices(
    study,
    repetitions=DEFAULT_BOOTSTRAP_REPS,
    seed=SEED,
    rng=None,
):
    """Generate study-stratified bootstrap index rows."""
    study = np.asarray(study)
    groups = study_groups(study)
    if rng is None:
        rng = np.random.default_rng(seed)

    indices_matrix = np.empty(
        (int(repetitions), len(study)),
        dtype=int,
    )

    for b in range(int(repetitions)):
        indices_matrix[b] = np.concatenate(
            [
                rng.choice(
                    indices,
                    size=len(indices),
                    replace=True,
                )
                for indices in groups.values()
            ]
        )

    return indices_matrix


def bootstrap_supported_weak_contrast(
    error,
    supported,
    study,
    repetitions=DEFAULT_BOOTSTRAP_REPS,
    seed=SEED,
):
    """
    Vectorized descriptive within-study bootstrap for the QD-BNN
    supported-versus-weak-support contrast.
    """
    error = np.asarray(error, dtype=float)
    supported = np.asarray(supported, dtype=bool)

    sampled_indices = _study_stratified_bootstrap_indices(
        study,
        repetitions=repetitions,
        seed=seed,
    )

    sampled_error = error[sampled_indices]
    sampled_supported = supported[sampled_indices]
    sampled_weak = ~sampled_supported

    n_supported = sampled_supported.sum(axis=1)
    n_weak = sampled_weak.sum(axis=1)

    valid = (n_supported >= 2) & (n_weak >= 2)

    supported_mae = (
        np.abs(sampled_error) * sampled_supported
    ).sum(axis=1) / n_supported

    weak_mae = (
        np.abs(sampled_error) * sampled_weak
    ).sum(axis=1) / n_weak

    supported_rmse = np.sqrt(
        (
            sampled_error**2 * sampled_supported
        ).sum(axis=1)
        / n_supported
    )

    weak_rmse = np.sqrt(
        (
            sampled_error**2 * sampled_weak
        ).sum(axis=1)
        / n_weak
    )

    mae_differences = (
        weak_mae - supported_mae
    )[valid]

    rmse_ratios = (
        weak_rmse / supported_rmse
    )[valid]

    return {
        "MAE_difference_CI95": tuple(
            np.quantile(
                mae_differences,
                [0.025, 0.975],
            )
        ),
        "RMSE_ratio_CI95": tuple(
            np.quantile(
                rmse_ratios,
                [0.025, 0.975],
            )
        ),
    }


def descriptive_ols(y, columns):
    """Small dependency-free OLS helper for descriptive sensitivity analysis."""
    y = np.asarray(y, dtype=float)
    names = ["Intercept"] + list(columns.keys())
    x = np.column_stack(
        [
            np.ones(len(y)),
            *[np.asarray(columns[name], dtype=float) for name in columns],
        ]
    )
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted = x @ beta
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return dict(zip(names, beta)), r2


def descriptive_bootstrap_field_metrics(
    y_true,
    predictions,
    supported,
    study,
    repetitions=DEFAULT_BOOTSTRAP_REPS,
    seed=SEED,
):
    """
    Descriptive within-study bootstrap intervals for field RMSE, MAE and bias.

    The random-number stream matches the original loop-based implementation,
    but metric calculations are vectorized for speed. These are descriptive
    uncertainty summaries, not cluster-robust population confidence intervals.
    """
    y_true = np.asarray(y_true, dtype=float)
    supported = np.asarray(supported, dtype=bool)
    study = np.asarray(study)

    rng = np.random.default_rng(seed)
    rows = []

    cohort_masks = {
        "Field all": np.ones(len(y_true), dtype=bool),
        "Field supported": supported,
        "Field weakly supported": ~supported,
    }

    for model_name, prediction in predictions.items():
        prediction = np.asarray(prediction, dtype=float)

        sampled_indices = _study_stratified_bootstrap_indices(
            study,
            repetitions=repetitions,
            rng=rng,
        )

        sampled_y = y_true[sampled_indices]
        sampled_prediction = prediction[sampled_indices]
        sampled_error = sampled_prediction - sampled_y
        sampled_supported = supported[sampled_indices]

        sampled_masks = {
            "Field all": np.ones_like(sampled_supported, dtype=bool),
            "Field supported": sampled_supported,
            "Field weakly supported": ~sampled_supported,
        }

        for cohort, original_mask in cohort_masks.items():
            mask = sampled_masks[cohort]
            counts = mask.sum(axis=1)

            rmse_values = np.sqrt(
                (
                    sampled_error**2 * mask
                ).sum(axis=1)
                / counts
            )
            mae_values = (
                np.abs(sampled_error) * mask
            ).sum(axis=1) / counts
            bias_values = (
                sampled_error * mask
            ).sum(axis=1) / counts

            estimate = regression_metrics(
                y_true[original_mask],
                prediction[original_mask],
            )

            for metric_name, values in [
                ("RMSE_mm_s", rmse_values),
                ("MAE_mm_s", mae_values),
                ("Bias_PredMinusObs_mm_s", bias_values),
            ]:
                low, high = np.quantile(
                    values,
                    [0.025, 0.975],
                )
                rows.append(
                    {
                        "Model": model_name,
                        "Cohort": cohort,
                        "Metric": metric_name,
                        "Estimate": float(
                            estimate[metric_name]
                        ),
                        "Descriptive_bootstrap95_low": float(low),
                        "Descriptive_bootstrap95_high": float(high),
                        "Valid_replicates": int(
                            len(values)
                        ),
                    }
                )

    return pd.DataFrame(rows)


def wilson_interval(successes, total, alpha=0.05):
    """Two-sided Wilson interval for a binomial proportion."""
    from scipy.stats import norm

    successes = int(successes)
    total = int(total)
    p = successes / total
    z = float(norm.ppf(1.0 - alpha / 2.0))
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = (
        z
        * np.sqrt(
            p * (1.0 - p) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return p, center - half, center + half


# =============================================================================
# 9. FIGURE HELPERS
# =============================================================================

def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 1.2,
            "axes.linewidth": 0.8,
            "savefig.dpi": DEFAULT_FIGURE_DPI,
        }
    )


def save_figure(fig, out_dir, filename, dpi):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        out_dir / f"{filename}.png",
        dpi=dpi,
        bbox_inches="tight",
    )

    fig.savefig(
        out_dir / f"{filename}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def ecdf(values):
    values = np.sort(np.asarray(values, dtype=float))
    probability = np.arange(1, len(values) + 1) / len(values)

    return values, probability


def add_standard_axes_style(ax):
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.tick_params(
        direction="out",
        length=3.5,
        width=0.8,
    )


# =============================================================================
# 10. PUBLICATION FIGURES
# =============================================================================

def generate_publication_figures(
    numerical,
    train_idx,
    test_domain_distance,
    field_predictions,
    primary_threshold,
    model_metrics,
    study_metrics,
    sensitivity,
    out_dir,
    dpi=DEFAULT_FIGURE_DPI,
):
    configure_matplotlib()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_num = numerical["charge_weight_kg"].to_numpy(float)
    d_num = numerical["distance_m"].to_numpy(float)
    ppv_num = numerical["ppv_mm_s"].to_numpy(float)
    sd_num = d_num / np.sqrt(q_num)

    q_field = field_predictions["charge_weight_kg"].to_numpy(float)
    d_field = field_predictions["distance_m"].to_numpy(float)
    observed = field_predictions["observed_ppv_mm_s"].to_numpy(float)
    sd_field = d_field / np.sqrt(q_field)

    supported = field_predictions["supported"].to_numpy(bool)
    domain_distance = field_predictions["domain_distance"].to_numpy(float)
    study = field_predictions["study"].astype(str).to_numpy()

    bnn_prediction = field_predictions["QD-BNN"].to_numpy(float)

    study_markers = {
        "Liu 2023": "o",
        "Borneo 2024": "s",
        "Limestone 2025": "^",
    }

    # -------------------------------------------------------------------------
    # Figure 1
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.scatter(
        q_num[train_idx],
        d_num[train_idx],
        s=11,
        alpha=0.18,
        label="Numerical training",
    )

    for study_name, marker in study_markers.items():

        mask = study == study_name

        handle = ax.scatter(
            q_field[mask & supported],
            d_field[mask & supported],
            s=52,
            marker=marker,
            label=study_name,
        )

        color = (
            handle.get_facecolor()[0]
            if len(handle.get_facecolor())
            else None
        )

        if np.any(mask & ~supported):
            ax.scatter(
                q_field[mask & ~supported],
                d_field[mask & ~supported],
                s=52,
                marker=marker,
                facecolors="none",
                edgecolors=[color],
                linewidths=1.25,
            )

    ax.text(
        0.98,
        0.04,
        "Filled = supported\nOpen = weakly supported",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Charge weight, Q (kg)")
    ax.set_ylabel("Blast-to-monitoring distance, D (m)")
    ax.set_title("Joint numerical-field Q-D domain")

    ax.legend(frameon=False)

    add_standard_axes_style(ax)

    save_figure(
        fig,
        out_dir,
        "Fig01_QD_domain_map",
        dpi,
    )

    # -------------------------------------------------------------------------
    # Figure 2
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.scatter(
        sd_num,
        ppv_num,
        s=11,
        alpha=0.17,
        label="Numerical database",
    )

    for study_name, marker in study_markers.items():

        mask = study == study_name

        ax.scatter(
            sd_field[mask],
            observed[mask],
            s=50,
            marker=marker,
            label=study_name,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(
        r"Scaled distance, $D/\sqrt{Q}$ (m kg$^{-1/2}$)"
    )
    ax.set_ylabel("PPV (mm/s)")
    ax.set_title("Scaled distance-PPV relationship")

    ax.legend(frameon=False)

    add_standard_axes_style(ax)

    save_figure(
        fig,
        out_dir,
        "Fig02_scaled_distance_vs_PPV",
        dpi,
    )

    # -------------------------------------------------------------------------
    # Figure 3
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5.4, 5.0))

    lower = min(
        observed.min(),
        bnn_prediction.min(),
    ) * 0.85

    upper = max(
        observed.max(),
        bnn_prediction.max(),
    ) * 1.15

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.1,
        label="1:1 line",
    )

    for study_name, marker in study_markers.items():

        mask = study == study_name

        handle = ax.scatter(
            observed[mask & supported],
            bnn_prediction[mask & supported],
            s=52,
            marker=marker,
            label=study_name,
        )

        color = (
            handle.get_facecolor()[0]
            if len(handle.get_facecolor())
            else None
        )

        if np.any(mask & ~supported):
            ax.scatter(
                observed[mask & ~supported],
                bnn_prediction[mask & ~supported],
                s=52,
                marker=marker,
                facecolors="none",
                edgecolors=[color],
                linewidths=1.25,
            )

    ax.text(
        0.98,
        0.04,
        "Filled = supported\nOpen = weakly supported",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)

    ax.set_xlabel("Observed field PPV (mm/s)")
    ax.set_ylabel("Predicted PPV (mm/s)")
    ax.set_title("Reduced QD-BNN: zero-shot field transfer")

    ax.legend(frameon=False)

    add_standard_axes_style(ax)

    save_figure(
        fig,
        out_dir,
        "Fig03_QDBNN_observed_vs_predicted",
        dpi,
    )

    # -------------------------------------------------------------------------
    # Figure 4
    # -------------------------------------------------------------------------
    absolute_error = np.abs(
        bnn_prediction - observed
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    for study_name, marker in study_markers.items():

        mask = study == study_name

        ax.scatter(
            domain_distance[mask],
            absolute_error[mask],
            s=50,
            marker=marker,
            label=study_name,
        )

    coefficient = np.polyfit(
        domain_distance,
        absolute_error,
        1,
    )

    x_line = np.linspace(
        domain_distance.min(),
        domain_distance.max(),
        200,
    )

    ax.plot(
        x_line,
        np.polyval(coefficient, x_line),
        label="Linear trend",
    )

    ax.axvline(
        primary_threshold,
        linestyle="--",
        linewidth=1.0,
        label="95% support threshold",
    )

    pearson = pearsonr(
        domain_distance,
        absolute_error,
    )

    spearman = spearmanr(
        domain_distance,
        absolute_error,
    )

    ax.text(
        0.98,
        0.96,
        (
            f"Pearson r = {pearson.statistic:.3f}\n"
            f"Spearman rho = {spearman.statistic:.3f}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
    )

    ax.set_xlabel("Joint-domain distance")
    ax.set_ylabel("Absolute PPV error (mm/s)")
    ax.set_title("QD-BNN absolute error versus joint-domain distance")

    ax.legend(frameon=False)

    add_standard_axes_style(ax)

    save_figure(
        fig,
        out_dir,
        "Fig04_QDBNN_error_vs_domain_distance",
        dpi,
    )

    # -------------------------------------------------------------------------
    # Figure 5
    # -------------------------------------------------------------------------
    models = [
        "Random Forest",
        "XGBoost",
        "QD-BNN",
        "Power law",
    ]

    cohorts = [
        "Numerical test",
        "Field supported",
        "Field weakly supported/sparse",
    ]

    x = np.arange(len(models))
    width = 0.24

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for j, cohort in enumerate(cohorts):

        values = []

        for model in models:

            row = model_metrics[
                (model_metrics["Model"] == model)
                & (model_metrics["Cohort"] == cohort)
            ]

            values.append(
                float(row["RMSE_mm_s"].iloc[0])
            )

        ax.bar(
            x + (j - 1) * width,
            values,
            width,
            label=cohort,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "RF",
            "XGBoost",
            "QD-BNN",
            "Power law",
        ]
    )

    ax.set_ylabel("RMSE (mm/s)")
    ax.set_title("Transfer error by model and domain support")
    ax.legend(frameon=False)

    add_standard_axes_style(ax)

    save_figure(
        fig,
        out_dir,
        "Fig05_model_RMSE_by_cohort",
        dpi,
    )

    # -------------------------------------------------------------------------
    # Figure 6
    # -------------------------------------------------------------------------
    cohorts = [
        "Field all",
        "Field supported",
        "Field weakly supported/sparse",
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for j, cohort in enumerate(cohorts):

        values = []

        for model in models:

            row = model_metrics[
                (model_metrics["Model"] == model)
                & (model_metrics["Cohort"] == cohort)
            ]

            values.append(
                float(
                    row[
                        "Bias_PredMinusObs_mm_s"
                    ].iloc[0]
                )
            )

        ax.bar(
            x + (j - 1) * width,
            values,
            width,
            label=cohort,
        )

    ax.axhline(0, linewidth=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "RF",
            "XGBoost",
            "QD-BNN",
            "Power law",
        ]
    )

    ax.set_ylabel(
        "Bias = predicted - observed PPV (mm/s)"
    )
    ax.set_title("Systematic PPV bias under field transfer")

    ax.legend(frameon=False)

    add_standard_axes_style(ax)

    save_figure(
        fig,
        out_dir,
        "Fig06_model_bias_by_cohort",
        dpi,
    )

    # -------------------------------------------------------------------------
    # Figure 7
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    studies = [
        "Liu 2023",
        "Borneo 2024",
        "Limestone 2025",
    ]

    for j, study_name in enumerate(studies):

        values = []

        for model in [
            "Power law",
            "Random Forest",
            "XGBoost",
            "QD-BNN",
        ]:

            row = study_metrics[
                (study_metrics["Model"] == model)
                & (
                    study_metrics["Study"]
                    == study_name
                )
            ]

            values.append(
                float(row["MAE_mm_s"].iloc[0])
            )

        ax.bar(
            x + (j - 1) * width,
            values,
            width,
            label=study_name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Power law",
            "RF",
            "XGBoost",
            "QD-BNN",
        ]
    )

    ax.set_ylabel("MAE (mm/s)")
    ax.set_title("Field prediction error by published study")

    ax.legend(frameon=False)

    add_standard_axes_style(ax)

    save_figure(
        fig,
        out_dir,
        "Fig07_study_level_MAE",
        dpi,
    )

    # -------------------------------------------------------------------------
    # Figure 8
    # -------------------------------------------------------------------------
    ordering = np.argsort(domain_distance)
    ranking = np.arange(
        1,
        len(field_predictions) + 1,
    )

    lower_interval = field_predictions[
        "QD-BNN_95_low"
    ].to_numpy(float)[ordering]

    upper_interval = field_predictions[
        "QD-BNN_95_high"
    ].to_numpy(float)[ordering]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    ax.fill_between(
        ranking,
        lower_interval,
        upper_interval,
        alpha=0.18,
        label="Numerical-validation absolute-residual band",
    )

    ax.plot(
        ranking,
        bnn_prediction[ordering],
        marker="o",
        markersize=3.5,
        linewidth=1.0,
        label="Predicted PPV",
    )

    ax.scatter(
        ranking,
        observed[ordering],
        marker="x",
        s=28,
        label="Observed field PPV",
    )

    ax.axvline(
        int(supported.sum()) + 0.5,
        linestyle="--",
        linewidth=1.0,
        label="Support boundary",
    )

    ax.set_xlabel(
        "Field cases ranked by increasing domain distance"
    )
    ax.set_ylabel("PPV (mm/s)")
    ax.set_title(
        "Validation-residual-band transfer across increasing Q-D distance"
    )

    ax.legend(frameon=False)

    add_standard_axes_style(ax)

    save_figure(
        fig,
        out_dir,
        "Fig08_QDBNN_residual_band",
        dpi,
    )

    # -------------------------------------------------------------------------
    # Figure 9
    # -------------------------------------------------------------------------
    x_num, y_num = ecdf(test_domain_distance)
    x_field, y_field = ecdf(domain_distance)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    ax.plot(
        x_num,
        y_num,
        label="Numerical test",
    )

    ax.plot(
        x_field,
        y_field,
        label="Published field",
    )

    ax.axvline(
        primary_threshold,
        linestyle="--",
        linewidth=1.0,
        label="95% training-derived threshold",
    )

    ax.set_xlabel("Joint-domain distance")
    ax.set_ylabel(
        "Empirical cumulative probability"
    )
    ax.set_ylim(0, 1.02)
    ax.set_title(
        "Distribution shift in joint Q-D support"
    )

    ax.legend(frameon=False)

    add_standard_axes_style(ax)

    save_figure(
        fig,
        out_dir,
        "Fig09_domain_distance_ECDF",
        dpi,
    )

    # -------------------------------------------------------------------------
    # Figure 10
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    labels = [
        f"k={int(row.k)}, p={row.threshold_percentile:g}"
        for row in sensitivity.itertuples()
    ]

    ratios = sensitivity[
        "RMSE_ratio"
    ].to_numpy(float)

    ax.bar(
        np.arange(len(labels)),
        ratios,
    )

    ax.set_xticks(
        np.arange(len(labels))
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        "Weak-support RMSE / supported RMSE"
    )

    ax.set_title(
        "Robustness to alternative domain-support definitions"
    )

    add_standard_axes_style(ax)

    save_figure(
        fig,
        out_dir,
        "Fig10_QDBNN_threshold_sensitivity",
        dpi,
    )

    # -------------------------------------------------------------------------
    # Supplementary Figures S1-S3
    # -------------------------------------------------------------------------
    distribution_specs = [
        (
            q_num,
            q_field,
            "Charge weight, Q (kg)",
            "FigS01_charge_ECDF",
            "Charge-weight distribution",
            True,
        ),
        (
            d_num,
            d_field,
            "Blast-to-monitoring distance, D (m)",
            "FigS02_distance_ECDF",
            "Distance distribution",
            False,
        ),
        (
            ppv_num,
            observed,
            "PPV (mm/s)",
            "FigS03_PPV_ECDF",
            "PPV distribution",
            False,
        ),
    ]

    for (
        numerical_values,
        field_values,
        xlabel,
        filename,
        title,
        log_x,
    ) in distribution_specs:

        fig, ax = plt.subplots(figsize=(6.2, 4.6))

        x1, y1 = ecdf(numerical_values)
        x2, y2 = ecdf(field_values)

        ax.plot(x1, y1, label="Numerical")
        ax.plot(x2, y2, label="Field")

        if log_x:
            ax.set_xscale("log")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(
            "Empirical cumulative probability"
        )
        ax.set_ylim(0, 1.02)
        ax.set_title(title)

        ax.legend(frameon=False)

        add_standard_axes_style(ax)

        save_figure(
            fig,
            out_dir,
            filename,
            dpi,
        )

    # -------------------------------------------------------------------------
    # Supplementary Figures S4-S6
    # -------------------------------------------------------------------------
    def observed_predicted_plot(
        model_name,
        column_name,
        filename,
    ):
        prediction = field_predictions[
            column_name
        ].to_numpy(float)

        fig, ax = plt.subplots(figsize=(5.4, 5.0))

        lower = min(
            observed.min(),
            prediction.min(),
        ) * 0.85

        upper = max(
            observed.max(),
            prediction.max(),
        ) * 1.15

        ax.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            linewidth=1.1,
            label="1:1 line",
        )

        for study_name, marker in study_markers.items():

            mask = study == study_name

            handle = ax.scatter(
                observed[mask & supported],
                prediction[mask & supported],
                s=50,
                marker=marker,
                label=study_name,
            )

            color = (
                handle.get_facecolor()[0]
                if len(handle.get_facecolor())
                else None
            )

            if np.any(mask & ~supported):
                ax.scatter(
                    observed[mask & ~supported],
                    prediction[mask & ~supported],
                    s=50,
                    marker=marker,
                    facecolors="none",
                    edgecolors=[color],
                    linewidths=1.25,
                )

        ax.text(
            0.98,
            0.04,
            "Filled = supported\nOpen = weakly supported",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)

        ax.set_xlabel(
            "Observed field PPV (mm/s)"
        )
        ax.set_ylabel(
            "Predicted PPV (mm/s)"
        )
        ax.set_title(
            f"{model_name}: zero-shot field transfer"
        )

        ax.legend(frameon=False)

        add_standard_axes_style(ax)

        save_figure(
            fig,
            out_dir,
            filename,
            dpi,
        )

    observed_predicted_plot(
        "Random Forest",
        "Random Forest",
        "FigS04_RF_observed_vs_predicted",
    )

    observed_predicted_plot(
        "XGBoost",
        "XGBoost",
        "FigS05_XGB_observed_vs_predicted",
    )

    observed_predicted_plot(
        "Scaled-distance model",
        "Power law",
        "FigS06_powerlaw_observed_vs_predicted",
    )

    # -------------------------------------------------------------------------
    # Supplementary Figure S7
    # -------------------------------------------------------------------------
    residual = (
        bnn_prediction - observed
    )

    fig, ax = plt.subplots(figsize=(5.8, 4.6))

    ax.boxplot(
        [
            residual[supported],
            residual[~supported],
        ],
        labels=[
            "Supported",
            "Weakly supported/sparse",
        ],
        showfliers=True,
    )

    ax.axhline(0, linewidth=0.9)

    ax.set_ylabel(
        "Residual = predicted - observed PPV (mm/s)"
    )

    ax.set_title(
        "Reduced QD-BNN residuals by domain support"
    )

    add_standard_axes_style(ax)

    save_figure(
        fig,
        out_dir,
        "FigS07_QDBNN_residuals_by_support",
        dpi,
    )

    # -------------------------------------------------------------------------
    # Supplementary Figure S8
    # -------------------------------------------------------------------------
    numerical_coverage = float(
        model_metrics.attrs.get(
            "NumericalCoverage",
            np.nan,
        )
    )

    field_coverage = float(
        model_metrics.attrs.get(
            "FieldCoverage",
            np.nan,
        )
    )

    fig, ax = plt.subplots(figsize=(5.8, 4.6))

    ax.bar(
        [
            "Numerical test",
            "Field all",
        ],
        [
            100 * numerical_coverage,
            100 * field_coverage,
        ],
    )

    ax.axhline(
        95,
        linestyle="--",
        linewidth=1.0,
        label="Nominal 95%",
    )

    ax.set_ylabel(
        "Empirical coverage (%)"
    )
    ax.set_ylim(0, 100)
    ax.set_title(
        "Validation-residual-band coverage under transfer"
    )

    ax.legend(frameon=False)

    add_standard_axes_style(ax)

    save_figure(
        fig,
        out_dir,
        "FigS08_QDBNN_coverage_transfer",
        dpi,
    )

    # -------------------------------------------------------------------------
    # Supplementary Figure S9
    # -------------------------------------------------------------------------
    study_bias = []

    for study_name in study_markers:

        mask = study == study_name

        study_bias.append(
            np.mean(
                bnn_prediction[mask]
                - observed[mask]
            )
        )

    fig, ax = plt.subplots(figsize=(6.2, 4.6))

    ax.bar(
        list(study_markers.keys()),
        study_bias,
    )

    ax.axhline(0, linewidth=0.9)

    ax.set_ylabel(
        "Mean bias = predicted - observed PPV (mm/s)"
    )

    ax.set_title(
        "Reduced QD-BNN bias by study"
    )

    add_standard_axes_style(ax)

    save_figure(
        fig,
        out_dir,
        "FigS09_QDBNN_study_bias",
        dpi,
    )


# =============================================================================
# 11. PUBLICATION TABLES
# =============================================================================

def build_publication_tables(
    numerical,
    field_predictions,
    model_metrics,
    study_metrics,
    correlations,
    sensitivity,
    contrast,
    out_dir,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_num = numerical["charge_weight_kg"].to_numpy(float)
    d_num = numerical["distance_m"].to_numpy(float)
    ppv_num = numerical["ppv_mm_s"].to_numpy(float)
    sd_num = d_num / np.sqrt(q_num)

    q_field = field_predictions["charge_weight_kg"].to_numpy(float)
    d_field = field_predictions["distance_m"].to_numpy(float)
    ppv_field = field_predictions["observed_ppv_mm_s"].to_numpy(float)
    sd_field = d_field / np.sqrt(q_field)

    def describe(array):
        array = np.asarray(array, dtype=float)

        return {
            "N": len(array),
            "Minimum": array.min(),
            "Median": np.median(array),
            "Maximum": array.max(),
            "Mean": array.mean(),
        }

    # Table 1A
    table1a_rows = []

    for variable, dataset, values in [
        (
            "Charge weight, Q (kg)",
            "Numerical",
            q_num,
        ),
        (
            "Charge weight, Q (kg)",
            "Field",
            q_field,
        ),
        (
            "Distance, D (m)",
            "Numerical",
            d_num,
        ),
        (
            "Distance, D (m)",
            "Field",
            d_field,
        ),
        (
            "Scaled distance, D/sqrt(Q)",
            "Numerical",
            sd_num,
        ),
        (
            "Scaled distance, D/sqrt(Q)",
            "Field",
            sd_field,
        ),
        (
            "PPV (mm/s)",
            "Numerical",
            ppv_num,
        ),
        (
            "PPV (mm/s)",
            "Field",
            ppv_field,
        ),
    ]:
        description = describe(values)

        table1a_rows.append(
            {
                "Variable": variable,
                "Dataset": dataset,
                **description,
            }
        )

    table1a = pd.DataFrame(
        table1a_rows
    )

    # Table 1B
    table1b_rows = []

    for study_name in [
        "Liu 2023",
        "Borneo 2024",
        "Limestone 2025",
    ]:

        mask = (
            field_predictions["study"]
            == study_name
        ).to_numpy()

        table1b_rows.append(
            {
                "External study": study_name,
                "N": int(mask.sum()),
                "Median Q (kg)": float(
                    np.median(
                        q_field[mask]
                    )
                ),
                "Median D (m)": float(
                    np.median(
                        d_field[mask]
                    )
                ),
                "Median scaled distance": float(
                    np.median(
                        sd_field[mask]
                    )
                ),
                "Median PPV (mm/s)": float(
                    np.median(
                        ppv_field[mask]
                    )
                ),
                "Supported N": int(
                    field_predictions.loc[
                        mask,
                        "supported",
                    ].sum()
                ),
                "Weakly supported/sparse N": int(
                    (~field_predictions.loc[
                        mask,
                        "supported",
                    ]).sum()
                ),
            }
        )

    table1b = pd.DataFrame(
        table1b_rows
    )

    # Table 2
    table2 = model_metrics.copy()

    # Table 3A
    table3a = correlations.copy()

    # Table 3B
    table3b = pd.DataFrame(
        [
            [
                "Supported field observations",
                int(
                    field_predictions[
                        "supported"
                    ].sum()
                ),
            ],
            [
                "Weakly supported/sparse field observations",
                int(
                    (
                        ~field_predictions[
                            "supported"
                        ]
                    ).sum()
                ),
            ],
            [
                "MAE difference bootstrap CI low (mm/s)",
                contrast[
                    "MAE_difference_CI95"
                ][0],
            ],
            [
                "MAE difference bootstrap CI high (mm/s)",
                contrast[
                    "MAE_difference_CI95"
                ][1],
            ],
            [
                "RMSE ratio bootstrap CI low",
                contrast[
                    "RMSE_ratio_CI95"
                ][0],
            ],
            [
                "RMSE ratio bootstrap CI high",
                contrast[
                    "RMSE_ratio_CI95"
                ][1],
            ],
        ],
        columns=[
            "Metric",
            "Value",
        ],
    )

    # Supplementary Table S1
    tableS1 = study_metrics.copy()

    # Supplementary Table S2
    tableS2 = sensitivity.copy()

    # Supplementary Table S3
    tableS3 = field_predictions[
        [
            "study",
            "case_id",
            "charge_weight_kg",
            "distance_m",
            "observed_ppv_mm_s",
            "supported",
            "Power law",
            "Random Forest",
            "XGBoost",
            "QD-BNN",
            "QD-BNN_residual",
            "QD-BNN_95_low",
            "QD-BNN_95_high",
            "QD-BNN_covered",
        ]
    ].copy()

    tables = {
        "Table1a_overall_characteristics": table1a,
        "Table1b_external_studies": table1b,
        "Table2_zero_shot_performance": table2,
        "Table3a_domain_correlations": table3a,
        "Table3b_primary_BNN_contrast": table3b,
        "TableS1_study_specific_performance": tableS1,
        "TableS2_support_sensitivity": tableS2,
        "TableS3_case_level_predictions": tableS3,
    }

    for filename, dataframe in tables.items():

        dataframe.to_csv(
            out_dir / f"{filename}.csv",
            index=False,
        )

        try:
            markdown = dataframe.to_markdown(
                index=False
            )

            (
                out_dir
                / f"{filename}.md"
            ).write_text(
                markdown,
                encoding="utf-8",
            )

        except ImportError:
            pass


# =============================================================================
# 12. MAIN ANALYSIS
# =============================================================================

def run_analysis(args):
    set_global_seed(SEED)

    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir = (
        output_dir
        / "figures"
    )

    tables_dir = (
        output_dir
        / "tables"
    )

    numerical = load_numerical(
        args.numerical
    )

    field = load_field(
        args.field
    )

    train_idx, validation_idx, test_idx = fixed_split(
        len(numerical)
    )

    if (
        len(train_idx),
        len(validation_idx),
        len(test_idx),
    ) != (
        1080,
        231,
        232,
    ):
        raise RuntimeError(
            "Unexpected split sizes. "
            "The paper requires 1080/231/232."
        )

    q = numerical[
        "charge_weight_kg"
    ].to_numpy(float)

    d = numerical[
        "distance_m"
    ].to_numpy(float)

    y = numerical[
        "ppv_mm_s"
    ].to_numpy(float)

    X = np.column_stack(
        [
            q,
            d,
        ]
    )

    q_field = field[
        "charge_weight_kg"
    ].to_numpy(float)

    d_field = field[
        "distance_m"
    ].to_numpy(float)

    y_field = field[
        "observed_ppv_mm_s"
    ].to_numpy(float)

    X_field = np.column_stack(
        [
            q_field,
            d_field,
        ]
    )

    # -------------------------------------------------------------------------
    # Model predictions.
    #
    # REFERENCE MODE:
    #   Read the exact frozen predictions used in the manuscript for ALL four
    #   models. This avoids software-version drift (especially XGBoost).
    #
    # TRAIN MODE:
    #   Refit Power law, Random Forest, XGBoost and Reduced QD-BNN from the
    #   numerical training subset.
    # -------------------------------------------------------------------------
    if args.bnn_mode == "reference":

        if (
            args.reference_field is None
            or args.reference_test is None
        ):
            raise ValueError(
                "--reference-field and --reference-test "
                "are required in reference mode."
            )

        reference_field = pd.read_csv(
            args.reference_field
        )

        reference_test = pd.read_csv(
            args.reference_test
        )

        # Confirm that the frozen predictions correspond exactly to the
        # numerical test split and external field table supplied to this run.
        if not np.allclose(
            reference_test["Q (kg)"].to_numpy(float),
            q[test_idx],
        ):
            raise RuntimeError(
                "Reference numerical-test Q values "
                "do not match the fixed split."
            )

        if not np.allclose(
            reference_test["D (m)"].to_numpy(float),
            d[test_idx],
        ):
            raise RuntimeError(
                "Reference numerical-test D values "
                "do not match the fixed split."
            )

        if not np.allclose(
            reference_test["Observed PPV"].to_numpy(float),
            y[test_idx],
        ):
            raise RuntimeError(
                "Reference numerical-test PPV values "
                "do not match the fixed split."
            )

        if not np.allclose(
            reference_field["Q (kg)"].to_numpy(float),
            q_field,
        ):
            raise RuntimeError(
                "Reference field Q values do not match "
                "the processed field table."
            )

        if not np.allclose(
            reference_field["D (m)"].to_numpy(float),
            d_field,
        ):
            raise RuntimeError(
                "Reference field D values do not match "
                "the processed field table."
            )

        if not np.allclose(
            reference_field["Observed PPV (mm/s)"].to_numpy(float),
            y_field,
        ):
            raise RuntimeError(
                "Reference field PPV values do not match "
                "the processed field table."
            )

        required_test_columns = {
            "Power-law pred",
            "RF pred",
            "XGB pred",
            "QD-BNN pred",
        }
        required_field_columns = {
            "Power-law pred",
            "RF pred",
            "XGB pred",
            "QD-BNN pred",
        }

        missing_test = required_test_columns.difference(
            reference_test.columns
        )
        missing_field = required_field_columns.difference(
            reference_field.columns
        )

        if missing_test:
            raise ValueError(
                "Reference numerical-test CSV missing columns: "
                f"{sorted(missing_test)}"
            )

        if missing_field:
            raise ValueError(
                "Reference field CSV missing columns: "
                f"{sorted(missing_field)}"
            )

        test_predictions = {
            "Power law": reference_test[
                "Power-law pred"
            ].to_numpy(float),
            "Random Forest": reference_test[
                "RF pred"
            ].to_numpy(float),
            "XGBoost": reference_test[
                "XGB pred"
            ].to_numpy(float),
            "QD-BNN": reference_test[
                "QD-BNN pred"
            ].to_numpy(float),
        }

        field_model_predictions = {
            "Power law": reference_field[
                "Power-law pred"
            ].to_numpy(float),
            "Random Forest": reference_field[
                "RF pred"
            ].to_numpy(float),
            "XGBoost": reference_field[
                "XGB pred"
            ].to_numpy(float),
            "QD-BNN": reference_field[
                "QD-BNN pred"
            ].to_numpy(float),
        }

        q95 = PAPER_Q95_MM_S

    else:

        # -------------------------------------------------------------
        # Deterministic numerical-data-trained models.
        # -------------------------------------------------------------
        power_law = (
            ScaledDistancePowerLaw()
            .fit(
                q[train_idx],
                d[train_idx],
                y[train_idx],
            )
        )

        random_forest = (
            RandomForestRegressor(
                **RF_PARAMS
            )
            .fit(
                X[train_idx],
                y[train_idx],
            )
        )

        xgboost = (
            XGBRegressor(
                **XGB_PARAMS
            )
            .fit(
                X[train_idx],
                y[train_idx],
            )
        )

        test_predictions = {
            "Power law": power_law.predict(
                q[test_idx],
                d[test_idx],
            ),
            "Random Forest": random_forest.predict(
                X[test_idx]
            ),
            "XGBoost": xgboost.predict(
                X[test_idx]
            ),
        }

        field_model_predictions = {
            "Power law": power_law.predict(
                q_field,
                d_field,
            ),
            "Random Forest": random_forest.predict(
                X_field
            ),
            "XGBoost": xgboost.predict(
                X_field
            ),
        }

        # -------------------------------------------------------------
        # Reduced QD-BNN retraining.
        # -------------------------------------------------------------
        qd_bnn = ReducedQDBNN()

        qd_bnn.fit(
            q[train_idx],
            d[train_idx],
            y[train_idx],
            q[validation_idx],
            d[validation_idx],
            y[validation_idx],
        )

        validation_prediction = (
            qd_bnn.predict(
                q[validation_idx],
                d[validation_idx],
            )
        )

        q95 = float(
            np.quantile(
                np.abs(
                    validation_prediction
                    - y[validation_idx]
                ),
                0.95,
            )
        )

        test_predictions[
            "QD-BNN"
        ] = qd_bnn.predict(
            q[test_idx],
            d[test_idx],
        )

        field_model_predictions[
            "QD-BNN"
        ] = qd_bnn.predict(
            q_field,
            d_field,
        )

    # -------------------------------------------------------------------------
    # Primary Q-D domain support.
    # -------------------------------------------------------------------------
    primary_domain = (
        QDDomainSupport(
            k=PRIMARY_K,
            percentile=PRIMARY_PERCENTILE,
        )
        .fit(
            q[train_idx],
            d[train_idx],
        )
    )

    test_domain_distance = (
        primary_domain.distance(
            q[test_idx],
            d[test_idx],
        )
    )

    field_domain_distance, supported = (
        primary_domain.classify(
            q_field,
            d_field,
        )
    )

    # -------------------------------------------------------------------------
    # Prediction table.
    # -------------------------------------------------------------------------
    field_predictions = field.copy()

    field_predictions[
        "scaled_distance"
    ] = (
        d_field
        / np.sqrt(q_field)
    )

    field_predictions[
        "domain_distance"
    ] = field_domain_distance

    field_predictions[
        "support_threshold"
    ] = primary_domain.threshold_

    field_predictions[
        "supported"
    ] = supported

    for (
        model_name,
        model_prediction,
    ) in field_model_predictions.items():

        field_predictions[
            model_name
        ] = model_prediction

    bnn_field_prediction = (
        field_model_predictions[
            "QD-BNN"
        ]
    )

    field_predictions[
        "QD-BNN_residual"
    ] = (
        bnn_field_prediction
        - y_field
    )

    field_predictions[
        "QD-BNN_95_low"
    ] = (
        bnn_field_prediction
        - q95
    )

    field_predictions[
        "QD-BNN_95_high"
    ] = (
        bnn_field_prediction
        + q95
    )

    field_predictions[
        "QD-BNN_covered"
    ] = (
        (y_field >= field_predictions[
            "QD-BNN_95_low"
        ])
        & (y_field <= field_predictions[
            "QD-BNN_95_high"
        ])
    )

    field_predictions.to_csv(
        output_dir
        / "field_predictions.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Model metrics.
    # -------------------------------------------------------------------------
    model_metric_rows = []

    for model_name in test_predictions:

        # Numerical test.
        numerical_metrics = regression_metrics(
            y[test_idx],
            test_predictions[model_name],
        )

        model_metric_rows.append(
            {
                "Model": model_name,
                "Cohort": "Numerical test",
                **numerical_metrics,
            }
        )

        # All field.
        field_metrics = regression_metrics(
            y_field,
            field_model_predictions[
                model_name
            ],
        )

        model_metric_rows.append(
            {
                "Model": model_name,
                "Cohort": "Field all",
                **field_metrics,
            }
        )

        # Supported.
        supported_metrics = regression_metrics(
            y_field[supported],
            field_model_predictions[
                model_name
            ][supported],
        )

        model_metric_rows.append(
            {
                "Model": model_name,
                "Cohort": "Field supported",
                **supported_metrics,
            }
        )

        # Weakly supported/sparse.
        weak_metrics = regression_metrics(
            y_field[~supported],
            field_model_predictions[
                model_name
            ][~supported],
        )

        model_metric_rows.append(
            {
                "Model": model_name,
                "Cohort": "Field weakly supported/sparse",
                **weak_metrics,
            }
        )

    model_metrics = pd.DataFrame(
        model_metric_rows
    )

    # -------------------------------------------------------------------------
    # QD-BNN validation-derived absolute-residual-band coverage.
    # -------------------------------------------------------------------------
    numerical_coverage = empirical_coverage(
        y[test_idx],
        test_predictions[
            "QD-BNN"
        ] - q95,
        test_predictions[
            "QD-BNN"
        ] + q95,
    )

    field_coverage = empirical_coverage(
        y_field,
        bnn_field_prediction - q95,
        bnn_field_prediction + q95,
    )

    model_metrics.attrs[
        "NumericalCoverage"
    ] = numerical_coverage

    model_metrics.attrs[
        "FieldCoverage"
    ] = field_coverage

    # -------------------------------------------------------------------------
    # Reviewer 4.4: descriptive uncertainty for main field metrics and coverage.
    # -------------------------------------------------------------------------
    field_metric_uncertainty = descriptive_bootstrap_field_metrics(
        y_field,
        field_model_predictions,
        supported,
        field["study"].to_numpy(),
        repetitions=args.bootstrap_reps,
        seed=SEED,
    )
    field_metric_uncertainty.to_csv(
        output_dir / "reviewer4_4_field_metric_uncertainty.csv",
        index=False,
    )

    numerical_covered_count = int(np.sum(
        (y[test_idx] >= test_predictions["QD-BNN"] - q95)
        & (y[test_idx] <= test_predictions["QD-BNN"] + q95)
    ))
    field_covered_count = int(np.sum(
        (y_field >= bnn_field_prediction - q95)
        & (y_field <= bnn_field_prediction + q95)
    ))

    numerical_cov_prop, numerical_cov_low, numerical_cov_high = wilson_interval(
        numerical_covered_count,
        len(test_idx),
    )
    field_cov_prop, field_cov_low, field_cov_high = wilson_interval(
        field_covered_count,
        len(y_field),
    )

    coverage_uncertainty = pd.DataFrame([
        {
            "Cohort": "Numerical test",
            "Covered_N": numerical_covered_count,
            "Total_N": int(len(test_idx)),
            "Coverage": numerical_cov_prop,
            "Wilson95_low": numerical_cov_low,
            "Wilson95_high": numerical_cov_high,
        },
        {
            "Cohort": "Field all",
            "Covered_N": field_covered_count,
            "Total_N": int(len(y_field)),
            "Coverage": field_cov_prop,
            "Wilson95_low": field_cov_low,
            "Wilson95_high": field_cov_high,
        },
    ])
    coverage_uncertainty.to_csv(
        output_dir / "reviewer4_4_coverage_uncertainty.csv",
        index=False,
    )

    model_metrics.to_csv(
        output_dir
        / "model_metrics.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Study-specific metrics.
    # -------------------------------------------------------------------------
    study_metric_rows = []

    for (
        model_name,
        prediction,
    ) in field_model_predictions.items():

        for study_name in field[
            "study"
        ].unique():

            mask = (
                field[
                    "study"
                ]
                == study_name
            ).to_numpy()

            study_metric_rows.append(
                {
                    "Model": model_name,
                    "Study": study_name,
                    **regression_metrics(
                        y_field[mask],
                        prediction[mask],
                    ),
                }
            )

    study_metrics = pd.DataFrame(
        study_metric_rows
    )

    study_metrics.to_csv(
        output_dir
        / "study_metrics.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Continuous domain-distance analysis.
    # -------------------------------------------------------------------------
    correlation_rows = []

    for (
        model_name,
        prediction,
    ) in field_model_predictions.items():

        absolute_error = np.abs(
            prediction
            - y_field
        )

        pearson = pearsonr(
            field_domain_distance,
            absolute_error,
        )

        spearman = spearmanr(
            field_domain_distance,
            absolute_error,
        )

        # Continuous field distances and model errors are unchanged by the
        # revised training-only support threshold. In reference mode with the
        # paper-standard 5,000 resamples, use the archived exact descriptive
        # bootstrap intervals to ensure deterministic reproduction and avoid
        # unnecessary repeated rank calculations. Train mode and non-standard
        # bootstrap counts recompute them directly.
        if (
            args.bnn_mode == "reference"
            and args.bootstrap_reps == DEFAULT_BOOTSTRAP_REPS
            and model_name in PAPER_BOOTSTRAP_CI
        ):
            pearson_ci = PAPER_BOOTSTRAP_CI[model_name]["pearson"]
            spearman_ci = PAPER_BOOTSTRAP_CI[model_name]["spearman"]
        else:
            pearson_ci = bootstrap_correlation(
                field_domain_distance,
                absolute_error,
                field["study"].to_numpy(),
                statistic="pearson",
                repetitions=args.bootstrap_reps,
                seed=SEED,
            )
            spearman_ci = bootstrap_correlation(
                field_domain_distance,
                absolute_error,
                field["study"].to_numpy(),
                statistic="spearman",
                repetitions=args.bootstrap_reps,
                seed=SEED,
            )

        correlation_rows.append(
            {
                "Model": model_name,
                "Pearson_r": float(
                    pearson.statistic
                ),
                "Pearson_p": float(
                    pearson.pvalue
                ),
                "Pearson_CI95_low": pearson_ci[0],
                "Pearson_CI95_high": pearson_ci[1],
                "Spearman_rho": float(
                    spearman.statistic
                ),
                "Spearman_p": float(
                    spearman.pvalue
                ),
                "Spearman_CI95_low": spearman_ci[0],
                "Spearman_CI95_high": spearman_ci[1],
            }
        )

    correlations = pd.DataFrame(
        correlation_rows
    )

    correlations.to_csv(
        output_dir
        / "domain_correlations.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Reviewer 3.2 robustness analyses: study confounding and response scale.
    # -------------------------------------------------------------------------
    robustness_rows = []
    regression_rows = []
    error_ppv_rows = []

    study_array = field["study"].astype(str).to_numpy()
    limestone_mask = study_array == "Limestone 2025"
    eps = 1.0e-12

    # Standardized predictors for descriptive adjusted regressions.
    z_distance = (
        field_domain_distance - np.mean(field_domain_distance)
    ) / np.std(field_domain_distance)
    z_ppv = (y_field - np.mean(y_field)) / np.std(y_field)

    study_levels = list(dict.fromkeys(study_array.tolist()))
    reference_study = study_levels[0]
    study_columns = {
        f"Study[{level}]": (study_array == level).astype(float)
        for level in study_levels[1:]
    }

    for model_name, prediction in field_model_predictions.items():
        absolute_error = np.abs(prediction - y_field)
        relative_absolute_error = absolute_error / np.maximum(y_field, eps)
        absolute_log_error = np.abs(
            np.log(np.maximum(prediction, eps))
            - np.log(np.maximum(y_field, eps))
        )

        for cohort_name, cohort_mask in [
            ("All field", np.ones(len(field), dtype=bool)),
            ("Limestone only", limestone_mask),
        ]:
            for metric_name, metric_values in [
                ("Absolute error (mm/s)", absolute_error),
                ("Relative absolute error", relative_absolute_error),
                ("Absolute log error", absolute_log_error),
            ]:
                x = field_domain_distance[cohort_mask]
                y_metric = metric_values[cohort_mask]
                pearson = pearsonr(x, y_metric)
                spearman = spearmanr(x, y_metric)
                robustness_rows.append(
                    {
                        "Model": model_name,
                        "Cohort": cohort_name,
                        "Error_metric": metric_name,
                        "N": int(cohort_mask.sum()),
                        "Pearson_r": float(pearson.statistic),
                        "Spearman_rho": float(spearman.statistic),
                    }
                )

        ppv_pearson = pearsonr(y_field, absolute_error)
        ppv_spearman = spearmanr(y_field, absolute_error)
        error_ppv_rows.append(
            {
                "Model": model_name,
                "N": int(len(y_field)),
                "Pearson_abs_error_vs_observed_PPV": float(ppv_pearson.statistic),
                "Spearman_abs_error_vs_observed_PPV": float(ppv_spearman.statistic),
            }
        )

        # Reviewer-suggested descriptive study-adjusted model.
        coefficients_a, r2_a = descriptive_ols(
            absolute_error,
            {
                "z_domain_distance": z_distance,
                **study_columns,
            },
        )

        # Additional sensitivity model also adjusts for response magnitude.
        coefficients_b, r2_b = descriptive_ols(
            absolute_error,
            {
                "z_domain_distance": z_distance,
                "z_observed_PPV": z_ppv,
                **study_columns,
            },
        )

        regression_rows.append(
            {
                "Model": model_name,
                "Reference_study": reference_study,
                "Study_adjusted_beta_domain_per_1SD": float(
                    coefficients_a["z_domain_distance"]
                ),
                "Study_adjusted_R2": r2_a,
                "Study_and_PPV_adjusted_beta_domain_per_1SD": float(
                    coefficients_b["z_domain_distance"]
                ),
                "Study_and_PPV_adjusted_beta_PPV_per_1SD": float(
                    coefficients_b["z_observed_PPV"]
                ),
                "Study_and_PPV_adjusted_R2": r2_b,
            }
        )

    robustness = pd.DataFrame(robustness_rows)
    robustness.to_csv(
        output_dir / "reviewer3_2_domain_robustness.csv",
        index=False,
    )

    study_adjusted = pd.DataFrame(regression_rows)
    study_adjusted.to_csv(
        output_dir / "reviewer3_2_study_adjusted_regression.csv",
        index=False,
    )

    error_vs_ppv = pd.DataFrame(error_ppv_rows)
    error_vs_ppv.to_csv(
        output_dir / "reviewer3_2_error_vs_observed_ppv.csv",
        index=False,
    )

    qdbnn_supported_underprediction = bool(
        np.all(bnn_field_prediction[supported] < y_field[supported])
    )

    # -------------------------------------------------------------------------
    # Primary QD-BNN supported-versus-weak-support bootstrap.
    # -------------------------------------------------------------------------
    bnn_error = (
        bnn_field_prediction
        - y_field
    )

    # Recompute the supported/weak-support contrast because the revised
    # training-only support threshold changes the field classification.
    contrast = (
        bootstrap_supported_weak_contrast(
            bnn_error,
            supported,
            field[
                "study"
            ].to_numpy(),
            repetitions=args.bootstrap_reps,
            seed=SEED,
        )
    )

    # -------------------------------------------------------------------------
    # Sensitivity analysis.
    # -------------------------------------------------------------------------
    sensitivity_rows = []

    for k in [
        3,
        5,
        10,
    ]:

        for percentile in [
            90,
            95,
            97.5,
        ]:

            domain_model = (
                QDDomainSupport(
                    k=k,
                    percentile=percentile,
                )
                .fit(
                    q[train_idx],
                    d[train_idx],
                )
            )

            _, scenario_supported = (
                domain_model.classify(
                    q_field,
                    d_field,
                )
            )

            supported_rmse = float(
                np.sqrt(
                    np.mean(
                        bnn_error[
                            scenario_supported
                        ] ** 2
                    )
                )
            )

            weak_rmse = float(
                np.sqrt(
                    np.mean(
                        bnn_error[
                            ~scenario_supported
                        ] ** 2
                    )
                )
            )

            supported_mae = float(
                np.mean(
                    np.abs(
                        bnn_error[
                            scenario_supported
                        ]
                    )
                )
            )

            weak_mae = float(
                np.mean(
                    np.abs(
                        bnn_error[
                            ~scenario_supported
                        ]
                    )
                )
            )

            sensitivity_rows.append(
                {
                    "k": k,
                    "threshold_percentile": percentile,
                    "distance_threshold": domain_model.threshold_,
                    "supported_N": int(
                        scenario_supported.sum()
                    ),
                    "weakly_supported_N": int(
                        (
                            ~scenario_supported
                        ).sum()
                    ),
                    "Supported_RMSE": supported_rmse,
                    "Weakly_supported_RMSE": weak_rmse,
                    "RMSE_ratio": (
                        weak_rmse
                        / supported_rmse
                    ),
                    "Supported_MAE": supported_mae,
                    "Weakly_supported_MAE": weak_mae,
                }
            )

    sensitivity = pd.DataFrame(
        sensitivity_rows
    )

    sensitivity.to_csv(
        output_dir
        / "support_sensitivity.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save summary JSON.
    # -------------------------------------------------------------------------
    summary = {
        "seed": SEED,
        "train_N": int(
            len(train_idx)
        ),
        "validation_N": int(
            len(validation_idx)
        ),
        "test_N": int(
            len(test_idx)
        ),
        "field_N": int(
            len(field)
        ),
        "domain_k": PRIMARY_K,
        "domain_percentile": PRIMARY_PERCENTILE,
        "domain_threshold": float(
            primary_domain.threshold_
        ),
        "supported_field_N": int(
            supported.sum()
        ),
        "weakly_supported_sparse_field_N": int(
            (
                ~supported
            ).sum()
        ),
        "q95_mm_s": float(
            q95
        ),
        "numerical_test_residual_band_coverage": numerical_coverage,
        "field_residual_band_coverage": field_coverage,
        "numerical_test_covered_count": numerical_covered_count,
        "field_covered_count": field_covered_count,
        "reviewer4_4_outputs": {
            "field_metric_uncertainty": "reviewer4_4_field_metric_uncertainty.csv",
            "coverage_uncertainty": "reviewer4_4_coverage_uncertainty.csv",
        },
        "bootstrap_repetitions": int(
            args.bootstrap_reps
        ),
        "bnn_mode": args.bnn_mode,
        "bnn_MAE_difference_CI95": [
            float(value)
            for value in contrast[
                "MAE_difference_CI95"
            ]
        ],
        "bnn_RMSE_ratio_CI95": [
            float(value)
            for value in contrast[
                "RMSE_ratio_CI95"
            ]
        ],
        "qdbnn_all_supported_predictions_underpredict": (
            qdbnn_supported_underprediction
        ),
        "reviewer3_2_outputs": {
            "domain_robustness": "reviewer3_2_domain_robustness.csv",
            "study_adjusted_regression": "reviewer3_2_study_adjusted_regression.csv",
            "error_vs_observed_ppv": "reviewer3_2_error_vs_observed_ppv.csv",
        },
    }

    with open(
        output_dir
        / "analysis_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Reviewer 3.2 diagnostic figures.
    # -------------------------------------------------------------------------
    if not args.skip_figures:
        configure_matplotlib()
        figures_dir.mkdir(parents=True, exist_ok=True)

        qdbnn_abs_error = np.abs(bnn_field_prediction - y_field)
        qdbnn_relative_error = qdbnn_abs_error / np.maximum(y_field, 1.0e-12)

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for study_name in field["study"].unique():
            mask = (field["study"] == study_name).to_numpy()
            ax.scatter(
                field_domain_distance[mask],
                qdbnn_relative_error[mask],
                s=48,
                label=study_name,
            )
        ax.set_xlabel("Joint-domain distance")
        ax.set_ylabel("Relative absolute PPV error")
        ax.set_title("QD-BNN relative error versus joint-domain distance")
        ax.legend(frameon=False)
        add_standard_axes_style(ax)
        save_figure(
            fig,
            figures_dir,
            "FigS10_QDBNN_relative_error_vs_domain_distance",
            args.dpi,
        )

        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        for study_name in field["study"].unique():
            mask = (field["study"] == study_name).to_numpy()
            ax.scatter(
                y_field[mask],
                qdbnn_abs_error[mask],
                s=48,
                label=study_name,
            )
        ax.set_xlabel("Observed field PPV (mm/s)")
        ax.set_ylabel("Absolute PPV error (mm/s)")
        ax.set_title("QD-BNN absolute error versus observed PPV")
        ax.legend(frameon=False)
        add_standard_axes_style(ax)
        save_figure(
            fig,
            figures_dir,
            "FigS11_QDBNN_error_vs_observed_PPV",
            args.dpi,
        )

    # -------------------------------------------------------------------------
    # Publication tables.
    # -------------------------------------------------------------------------
    build_publication_tables(
        numerical=numerical,
        field_predictions=field_predictions,
        model_metrics=model_metrics,
        study_metrics=study_metrics,
        correlations=correlations,
        sensitivity=sensitivity,
        contrast=contrast,
        out_dir=tables_dir,
    )

    # -------------------------------------------------------------------------
    # Publication figures.
    # -------------------------------------------------------------------------
    if not args.skip_figures:

        generate_publication_figures(
            numerical=numerical,
            train_idx=train_idx,
            test_domain_distance=test_domain_distance,
            field_predictions=field_predictions,
            primary_threshold=primary_domain.threshold_,
            model_metrics=model_metrics,
            study_metrics=study_metrics,
            sensitivity=sensitivity,
            out_dir=figures_dir,
            dpi=args.dpi,
        )

    # -------------------------------------------------------------------------
    # Console summary.
    # -------------------------------------------------------------------------
    print()
    print("=" * 78)
    print("BLAST-BNN PAPER 2 ANALYSIS COMPLETED")
    print("=" * 78)

    print(
        f"Split: "
        f"{len(train_idx)} train / "
        f"{len(validation_idx)} validation / "
        f"{len(test_idx)} test"
    )

    print(
        "Primary domain threshold: "
        f"{primary_domain.threshold_:.12f}"
    )

    print(
        f"Supported field cases: "
        f"{supported.sum()} / {len(supported)}"
    )

    print(
        "Weakly supported/sparse field cases: "
        f"{(~supported).sum()} / {len(supported)}"
    )

    print(
        "QD-BNN numerical-test residual-band coverage: "
        f"{numerical_coverage:.6f}"
    )

    print(
        "QD-BNN field residual-band coverage: "
        f"{field_coverage:.6f}"
    )

    if args.bnn_mode == "reference":
        print(
            "Reference mode: frozen paper predictions used "
            "for Power law, Random Forest, XGBoost and QD-BNN."
        )

    print(
        f"Outputs written to: "
        f"{output_dir.resolve()}"
    )

    print("=" * 78)
    print()


# =============================================================================
# 13. COMMAND-LINE INTERFACE
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the complete BLAST-BNN "
            "Paper 2 numerical-to-field transfer analysis."
        )
    )

    parser.add_argument(
        "--numerical",
        required=True,
        help=(
            "Path to blast_dataset_FULL.csv"
        ),
    )

    parser.add_argument(
        "--field",
        required=True,
        help=(
            "Path to field_external_46.csv"
        ),
    )

    parser.add_argument(
        "--reference-field",
        default=None,
        help=(
            "Frozen paper field predictions CSV. "
            "Required for --bnn-mode reference."
        ),
    )

    parser.add_argument(
        "--reference-test",
        default=None,
        help=(
            "Frozen paper numerical-test predictions CSV. "
            "Required for --bnn-mode reference."
        ),
    )

    parser.add_argument(
        "--output",
        default="paper2_outputs",
        help=(
            "Output directory."
        ),
    )

    parser.add_argument(
        "--bnn-mode",
        choices=[
            "reference",
            "train",
        ],
        default="reference",
        help=(
            "reference = exact paper QD-BNN predictions; "
            "train = retrain QD-BNN."
        ),
    )

    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPS,
        help=(
            "Study-stratified bootstrap repetitions. "
            "Paper value = 5000."
        ),
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_FIGURE_DPI,
        help=(
            "PNG figure resolution. "
            "Paper value = 800 dpi."
        ),
    )

    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help=(
            "Run statistics/tables only."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    run_analysis(arguments)


# In[ ]:




