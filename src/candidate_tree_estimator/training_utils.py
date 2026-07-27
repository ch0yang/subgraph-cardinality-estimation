from __future__ import annotations

import math
import zlib

import numpy as np
import pandas as pd
import torch


VALIDATION_FRACTION_OF_OUTER_TRAIN = 0.125
NATIVE_SCALE_FEATURE_INDICES = (9, 10)
TAIL_THRESHOLDS = (1e2, 1e3, 1e4, 1e5, 1e6, 1e7)


def condition_values(query_sizes: np.ndarray) -> np.ndarray:
    return (
        np.log1p(query_sizes.astype(np.float64)) / math.log(33.0)
    ).reshape(-1, 1)


def _fit_line(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(design, y, rcond=None)[0]


def _predict_line(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    return coefficients[0] + coefficients[1] * x


def normalize_fold(
    train_raw: np.ndarray,
    eval_raw: np.ndarray,
    train_sizes: np.ndarray,
    eval_sizes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | bool]]]:
    train_out = np.zeros_like(train_raw, dtype=np.float64)
    eval_out = np.zeros_like(eval_raw, dtype=np.float64)
    log_train_size = np.log1p(train_sizes.astype(np.float64))
    log_eval_size = np.log1p(eval_sizes.astype(np.float64))
    unique_sizes = np.asarray(sorted(np.unique(train_sizes)), dtype=int)
    diagnostics: list[dict[str, float | int | bool]] = []

    for feature_index in range(train_raw.shape[1]):
        if feature_index in NATIVE_SCALE_FEATURE_INDICES:
            train_out[:, feature_index] = train_raw[:, feature_index]
            eval_out[:, feature_index] = eval_raw[:, feature_index]
            diagnostics.append(
                {
                    "feature_index": feature_index,
                    "native_scale": True,
                }
            )
            continue

        train_values = train_raw[:, feature_index]
        global_mean = float(train_values.mean())
        global_std = float(train_values.std())
        if global_std < 1e-8:
            global_std = 1.0

        if len(unique_sizes) >= 2:
            group_x = np.log1p(unique_sizes.astype(np.float64))
            group_mean = np.asarray(
                [
                    train_values[train_sizes == query_size].mean()
                    for query_size in unique_sizes
                ],
                dtype=np.float64,
            )
            group_std = np.asarray(
                [
                    train_values[train_sizes == query_size].std()
                    for query_size in unique_sizes
                ],
                dtype=np.float64,
            )
            group_std = np.maximum(group_std, 0.05 * global_std)
            mean_coefficients = _fit_line(group_x, group_mean)
            log_std_coefficients = _fit_line(
                group_x,
                np.log(group_std),
            )
            train_mean = _predict_line(
                mean_coefficients,
                log_train_size,
            )
            eval_mean = _predict_line(
                mean_coefficients,
                log_eval_size,
            )
            train_scale = np.exp(
                _predict_line(log_std_coefficients, log_train_size)
            )
            eval_scale = np.exp(
                _predict_line(log_std_coefficients, log_eval_size)
            )
        else:
            mean_coefficients = np.asarray([global_mean, 0.0])
            log_std_coefficients = np.asarray(
                [math.log(global_std), 0.0]
            )
            train_mean = np.full(len(train_raw), global_mean)
            eval_mean = np.full(len(eval_raw), global_mean)
            train_scale = np.full(len(train_raw), global_std)
            eval_scale = np.full(len(eval_raw), global_std)

        minimum_scale = 0.25 * global_std
        maximum_scale = 4.0 * global_std
        train_scale = np.clip(
            train_scale,
            minimum_scale,
            maximum_scale,
        )
        eval_scale = np.clip(
            eval_scale,
            minimum_scale,
            maximum_scale,
        )
        train_out[:, feature_index] = (
            train_values - train_mean
        ) / train_scale
        eval_out[:, feature_index] = (
            eval_raw[:, feature_index] - eval_mean
        ) / eval_scale
        diagnostics.append(
            {
                "feature_index": feature_index,
                "native_scale": False,
                "global_mean": global_mean,
                "global_std": global_std,
                "mean_intercept": float(mean_coefficients[0]),
                "mean_log_size_slope": float(mean_coefficients[1]),
                "log_std_intercept": float(log_std_coefficients[0]),
                "log_std_log_size_slope": float(
                    log_std_coefficients[1]
                ),
            }
        )
    return train_out, eval_out, diagnostics


def stable_validation_indices(
    frame: pd.DataFrame,
    outer_fold: int,
    validation_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folds = frame["fold"].to_numpy(dtype=int)
    test_index = np.flatnonzero(folds == outer_fold)
    outer_train_index = np.flatnonzero(folds != outer_fold)
    query_sizes = frame["query_size"].to_numpy(dtype=int)
    query_names = frame["query_name"].astype(str).to_numpy()
    dataset = str(frame["dataset"].iloc[0])
    validation_parts: list[np.ndarray] = []

    for query_size in sorted(set(query_sizes[outer_train_index])):
        candidates = outer_train_index[
            query_sizes[outer_train_index] == query_size
        ]
        requested = int(
            round(
                len(candidates)
                * VALIDATION_FRACTION_OF_OUTER_TRAIN
            )
        )
        validation_count = min(
            max(1, requested),
            max(1, len(candidates) - 1),
        )
        scores = np.asarray(
            [
                zlib.crc32(
                    (
                        f"{dataset}|{query_size}|{query_names[index]}|"
                        f"{outer_fold}|{validation_seed}"
                    ).encode("utf-8")
                )
                for index in candidates
            ],
            dtype=np.uint32,
        )
        order = np.lexsort((query_names[candidates], scores))
        validation_parts.append(
            np.sort(candidates[order[:validation_count]])
        )

    validation_index = np.sort(np.concatenate(validation_parts))
    train_index = np.setdiff1d(
        outer_train_index,
        validation_index,
        assume_unique=True,
    )
    if np.intersect1d(train_index, validation_index).size:
        raise RuntimeError("train/validation overlap")
    if np.intersect1d(test_index, outer_train_index).size:
        raise RuntimeError("outer train/test overlap")
    if (
        len(train_index)
        + len(validation_index)
        + len(test_index)
        != len(frame)
    ):
        raise RuntimeError("split coverage mismatch")
    return train_index, validation_index, test_index


def size_balanced_weights(
    frame: pd.DataFrame,
    indices: np.ndarray,
) -> torch.Tensor:
    query_sizes = frame.iloc[indices]["query_size"].to_numpy(dtype=int)
    unique_sizes = sorted(int(value) for value in np.unique(query_sizes))
    weights = np.zeros(len(indices), dtype=np.float64)
    target_mass = len(indices) / max(len(unique_sizes), 1)
    for query_size in unique_sizes:
        mask = query_sizes == query_size
        weights[mask] = target_mass / max(int(mask.sum()), 1)
    return torch.tensor(weights, dtype=torch.float32)


def qerror_from_logs(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    absolute_log10 = np.abs(predictions - targets) / math.log(10.0)
    return np.power(
        10.0,
        np.clip(absolute_log10, 0.0, 300.0),
    )


def summarize_qerror(values: np.ndarray) -> dict[str, float | int]:
    qerrors = np.asarray(values, dtype=np.float64)
    return {
        "rows": int(len(qerrors)),
        "q50": float(np.quantile(qerrors, 0.50)),
        "q95": float(np.quantile(qerrors, 0.95)),
        **{
            f"gt_{threshold:.0e}": int(np.sum(qerrors > threshold))
            for threshold in TAIL_THRESHOLDS
        },
        "max_qerror": float(np.max(qerrors)),
    }
