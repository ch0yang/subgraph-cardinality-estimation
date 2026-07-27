from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from candidate_tree_estimator.features import FEATURES
from candidate_tree_estimator.model import build_estimator
from candidate_tree_estimator.training_utils import (
    condition_values,
    normalize_fold,
    qerror_from_logs,
    size_balanced_weights,
    stable_validation_indices,
    summarize_qerror,
)


EXPECTED_PARAMETERS = 4677


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".pkl" or path.suffix == ".pickle":
        return pd.read_pickle(path)
    if path.name.endswith(".csv.gz") or path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported feature-frame format: {path}")


def configure_frame(frame: pd.DataFrame) -> pd.DataFrame:
    configured = frame.copy()
    configured["dataset"] = configured["dataset"].astype(str).str.lower()
    configured["query_size"] = configured["query_size"].astype(int)
    configured["fold"] = configured["fold"].astype(int)
    if "query_name" not in configured.columns:
        if "query" not in configured.columns:
            raise ValueError("frame has neither query_name nor query")
        configured["query_name"] = configured["query"].astype(str)
    configured["query_name"] = configured["query_name"].astype(str)
    required = set(FEATURES) | {
        "dataset",
        "query_size",
        "query_name",
        "fold",
        "tree_log",
        "target_log",
    }
    missing = sorted(required - set(configured.columns))
    if missing:
        raise ValueError(f"feature frame lacks columns: {missing}")
    key = ["dataset", "query_size", "query_name"]
    if configured.duplicated(key).any():
        raise ValueError("duplicate query identifiers")
    if sorted(configured["fold"].unique().tolist()) != list(range(5)):
        raise ValueError("expected outer folds 0 through 4")
    if not np.isfinite(
        configured[FEATURES].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("feature frame contains non-finite values")
    return configured


def build_model(seed: int) -> torch.nn.Module:
    model = build_estimator(seed=seed)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )
    if parameter_count != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"parameter-count mismatch: {parameter_count}"
        )
    return model


def prepare_arrays(
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.nan_to_num(
        frame[FEATURES].to_numpy(dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    targets = frame["target_log"].to_numpy(dtype=np.float64)
    tree = frame["tree_log"].to_numpy(dtype=np.float64)
    query_sizes = frame["query_size"].to_numpy(dtype=int)
    conditions = condition_values(query_sizes)
    return features, targets, tree, query_sizes, conditions


def predict_logs(
    model: torch.nn.Module,
    features: torch.Tensor,
    condition: torch.Tensor,
    tree_log: torch.Tensor,
) -> torch.Tensor:
    if not hasattr(model, "forward_logit"):
        raise TypeError("model does not expose forward_logit")
    ratio = torch.sigmoid(
        model.forward_logit(features, condition)
    )
    return tree_log * (1.0 - ratio)


def row_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return (
        F.softplus(predictions) - F.softplus(targets)
    ).square()


def select_epoch(
    frame: pd.DataFrame,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    max_epochs: int,
    model_seed: int,
) -> tuple[int, pd.DataFrame]:
    features, targets, tree, query_sizes, conditions = prepare_arrays(frame)
    train_x, validation_x, _ = normalize_fold(
        features[train_index],
        features[validation_index],
        query_sizes[train_index],
        query_sizes[validation_index],
    )
    model = build_model(model_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3,
    )
    x_train = torch.tensor(train_x, dtype=torch.float32)
    x_validation = torch.tensor(validation_x, dtype=torch.float32)
    condition_train = torch.tensor(
        conditions[train_index],
        dtype=torch.float32,
    )
    condition_validation = torch.tensor(
        conditions[validation_index],
        dtype=torch.float32,
    )
    target_train = torch.tensor(
        targets[train_index],
        dtype=torch.float32,
    )
    target_validation = torch.tensor(
        targets[validation_index],
        dtype=torch.float32,
    )
    tree_train = torch.tensor(
        tree[train_index],
        dtype=torch.float32,
    )
    tree_validation = torch.tensor(
        tree[validation_index],
        dtype=torch.float32,
    )
    train_weight = size_balanced_weights(frame, train_index)
    validation_weight = size_balanced_weights(
        frame,
        validation_index,
    )

    curve: list[dict[str, float | int | bool]] = []
    best_epoch = -1
    best_validation_loss = math.inf
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_predictions = predict_logs(
            model,
            x_train,
            condition_train,
            tree_train,
        )
        train_loss = (
            row_loss(train_predictions, target_train)
            * train_weight
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_predictions = predict_logs(
                model,
                x_validation,
                condition_validation,
                tree_validation,
            )
            validation_loss = (
                row_loss(
                    validation_predictions,
                    target_validation,
                )
                * validation_weight
            ).mean()
        validation_loss_value = float(validation_loss.item())
        validation_values = (
            validation_predictions.cpu().numpy().astype(np.float64)
        )
        validation_qerror = qerror_from_logs(
            validation_values,
            targets[validation_index],
        )
        selected = validation_loss_value < best_validation_loss
        if selected:
            best_validation_loss = validation_loss_value
            best_epoch = epoch
        curve.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.item()),
                "validation_loss": validation_loss_value,
                "validation_q50": float(
                    np.quantile(validation_qerror, 0.50)
                ),
                "validation_q95": float(
                    np.quantile(validation_qerror, 0.95)
                ),
                "validation_max_qerror": float(
                    np.max(validation_qerror)
                ),
                "best_so_far": selected,
            }
        )
    if best_epoch <= 0:
        raise RuntimeError("no validation checkpoint selected")
    return best_epoch, pd.DataFrame(curve)


def retrain_and_predict(
    frame: pd.DataFrame,
    train_index: np.ndarray,
    test_index: np.ndarray,
    epochs: int,
    model_seed: int,
) -> np.ndarray:
    features, targets, tree, query_sizes, conditions = prepare_arrays(frame)
    train_x, test_x, _ = normalize_fold(
        features[train_index],
        features[test_index],
        query_sizes[train_index],
        query_sizes[test_index],
    )
    model = build_model(model_seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3,
    )
    x_train = torch.tensor(train_x, dtype=torch.float32)
    x_test = torch.tensor(test_x, dtype=torch.float32)
    condition_train = torch.tensor(
        conditions[train_index],
        dtype=torch.float32,
    )
    condition_test = torch.tensor(
        conditions[test_index],
        dtype=torch.float32,
    )
    target_train = torch.tensor(
        targets[train_index],
        dtype=torch.float32,
    )
    tree_train = torch.tensor(
        tree[train_index],
        dtype=torch.float32,
    )
    tree_test = torch.tensor(
        tree[test_index],
        dtype=torch.float32,
    )
    train_weight = size_balanced_weights(frame, train_index)

    for _ in range(epochs):
        model.train()
        predictions = predict_logs(
            model,
            x_train,
            condition_train,
            tree_train,
        )
        loss = (
            row_loss(predictions, target_train)
            * train_weight
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        return (
            predict_logs(
                model,
                x_test,
                condition_test,
                tree_test,
            )
            .cpu()
            .numpy()
            .astype(np.float64)
        )


def run_fold(
    frame: pd.DataFrame,
    outer_fold: int,
    max_epochs: int,
    seed_base: int,
    validation_seed: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    train_index, validation_index, test_index = (
        stable_validation_indices(
            frame,
            outer_fold,
            validation_seed,
        )
    )
    model_seed = seed_base + outer_fold
    started = time.time()
    selected_epoch, curve = select_epoch(
        frame,
        train_index,
        validation_index,
        max_epochs,
        model_seed,
    )
    full_train_index = np.sort(
        np.concatenate([train_index, validation_index])
    )
    test_predictions = retrain_and_predict(
        frame,
        full_train_index,
        test_index,
        selected_epoch,
        model_seed,
    )
    targets = frame["target_log"].to_numpy(dtype=np.float64)[
        test_index
    ]
    qerrors = qerror_from_logs(test_predictions, targets)
    rows = pd.DataFrame(
        {
            "dataset": frame.iloc[test_index]["dataset"].to_numpy(),
            "query_size": frame.iloc[test_index][
                "query_size"
            ].to_numpy(),
            "query_name": frame.iloc[test_index][
                "query_name"
            ].to_numpy(),
            "fold": frame.iloc[test_index]["fold"].to_numpy(dtype=int),
            "outer_fold": outer_fold,
            "selected_epoch": selected_epoch,
            "target_log": targets,
            "pred_log": test_predictions,
            "signed_log10_qerror": (
                test_predictions - targets
            )
            / math.log(10.0),
            "qerror": qerrors,
        }
    )
    curve.insert(0, "dataset", str(frame["dataset"].iloc[0]))
    curve.insert(1, "outer_fold", outer_fold)
    curve["selected"] = curve["epoch"].eq(selected_epoch)
    selected_row = curve.loc[curve["selected"]].iloc[0]
    summary = {
        "dataset": str(frame["dataset"].iloc[0]),
        "outer_fold": outer_fold,
        "train_rows_for_selection": len(train_index),
        "validation_rows": len(validation_index),
        "retrain_rows": len(full_train_index),
        "test_rows": len(test_index),
        "selected_epoch": selected_epoch,
        "selected_validation_loss": float(
            selected_row["validation_loss"]
        ),
        "selected_validation_q50": float(
            selected_row["validation_q50"]
        ),
        "selected_validation_q95": float(
            selected_row["validation_q95"]
        ),
        **summarize_qerror(qerrors),
        "elapsed_seconds": time.time() - started,
    }
    return summary, rows, curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--validation-seed", type=int, default=20260722)
    parser.add_argument("--expected-rows", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    if args.out_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite: {args.out_dir}"
        )
    args.out_dir.mkdir(parents=True)
    torch.set_num_threads(args.torch_threads)
    frame = configure_frame(load_frame(args.frame))
    if args.expected_rows and len(frame) != args.expected_rows:
        raise RuntimeError(
            f"row mismatch: {len(frame)} != {args.expected_rows}"
        )

    summaries = []
    row_frames = []
    curve_frames = []
    for dataset in sorted(frame["dataset"].unique()):
        scoped = frame.loc[
            frame["dataset"].eq(dataset)
        ].reset_index(drop=True)
        for outer_fold in range(5):
            summary, rows, curve = run_fold(
                scoped,
                outer_fold,
                args.max_epochs,
                args.seed_base,
                args.validation_seed,
            )
            summaries.append(summary)
            row_frames.append(rows)
            curve_frames.append(curve)
            print(json.dumps(summary), flush=True)

    summary_frame = pd.DataFrame(summaries).sort_values(
        ["dataset", "outer_fold"]
    )
    rows = pd.concat(row_frames, ignore_index=True).sort_values(
        ["dataset", "query_size", "query_name"]
    ).reset_index(drop=True)
    curves = pd.concat(curve_frames, ignore_index=True).sort_values(
        ["dataset", "outer_fold", "epoch"]
    ).reset_index(drop=True)
    if len(rows) != len(frame):
        raise RuntimeError("outer-test prediction coverage mismatch")
    key = ["dataset", "query_size", "query_name"]
    if rows.duplicated(key).any():
        raise RuntimeError("duplicate outer-test predictions")

    query_sets = (
        rows.groupby(["dataset", "query_size"])["qerror"]
        .agg(
            set_q50=lambda values: values.quantile(0.50),
            set_q95=lambda values: values.quantile(0.95),
            max_qerror="max",
        )
        .reset_index()
    )
    aggregate = {
        **summarize_qerror(rows["qerror"].to_numpy(dtype=float)),
        "query_graph_sets": len(query_sets),
        "macro_geomean_q50": float(
            np.exp(np.log(query_sets["set_q50"]).mean())
        ),
        "macro_geomean_q95": float(
            np.exp(np.log(query_sets["set_q95"]).mean())
        ),
        "selected_epoch_min": int(
            summary_frame["selected_epoch"].min()
        ),
        "selected_epoch_median": float(
            summary_frame["selected_epoch"].median()
        ),
        "selected_epoch_max": int(
            summary_frame["selected_epoch"].max()
        ),
        "selected_at_max_epochs": int(
            summary_frame["selected_epoch"]
            .eq(args.max_epochs)
            .sum()
        ),
    }
    summary_frame.to_csv(
        args.out_dir / "model_selection_summary.csv",
        index=False,
    )
    rows.to_csv(
        args.out_dir / "outer_test_rows.csv.gz",
        index=False,
    )
    curves.to_csv(
        args.out_dir / "validation_curves.csv.gz",
        index=False,
    )
    query_sets.to_csv(
        args.out_dir / "query_graph_set_summary.csv",
        index=False,
    )
    (args.out_dir / "outer_test_summary.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "data_split": (
            "five outer folds; 20% test; the remaining 80% split "
            "by query size into approximately 70% train and 10% "
            "validation"
        ),
        "features": FEATURES,
        "groups": [[0, 1, 2], [3, 4], [5, 6], [7, 8, 9, 10]],
        "candidate_tree_strategy": "mwst_low_density_edges",
        "gql_refinement_rounds": 5,
        "tree_probe_budget": 512,
        "graph_probe_budget": 512,
        "normalization": (
            "training-fitted query-size trends; feature indices "
            "9 and 10 remain on native scales"
        ),
        "loss": "uniform query-size-balanced count-domain MSLE",
        "selected_hyperparameter": "training epoch only",
        "checkpoint_candidates": (
            f"every epoch from 1 to {args.max_epochs}"
        ),
        "retraining": (
            "reinitialize and train on the complete 80% "
            "outer-training partition"
        ),
        "seed_base": args.seed_base,
        "validation_seed": args.validation_seed,
        "trainable_parameters": EXPECTED_PARAMETERS,
        "mlp_is_final_predictor": True,
        "post_mlp_replacement": False,
        "post_mlp_blend": False,
        "post_mlp_routing": False,
        "post_mlp_clamp": False,
        "summary": aggregate,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
