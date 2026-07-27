from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from candidate_tree_estimator.features import (
    FEATURES,
    KEY,
    load_export_jsonl,
    load_ground_truth,
)


def assign_folds(
    frame: pd.DataFrame,
    fold_count: int,
    fold_seed: int,
) -> pd.Series:
    folds = pd.Series(index=frame.index, dtype=int)
    for _, group in frame.groupby(["dataset", "query_size"], sort=True):
        if len(group) < fold_count:
            raise ValueError(
                "each dataset/query-size group needs at least "
                f"{fold_count} rows; observed {len(group)}"
            )
        scores = np.asarray(
            [
                zlib.crc32(
                    (
                        f"{row.dataset}|{row.query_size}|{row.query_name}|"
                        f"{fold_seed}"
                    ).encode("utf-8")
                )
                for row in group.itertuples()
            ],
            dtype=np.uint32,
        )
        names = group["query_name"].to_numpy(dtype=str)
        order = np.lexsort((names, scores))
        ordered_index = group.index.to_numpy()[order]
        folds.loc[ordered_index] = np.arange(len(group)) % fold_count
    return folds.astype(int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=42)
    args = parser.parse_args()

    if args.fold_count != 5:
        raise ValueError("the current training runner expects five folds")

    features = load_export_jsonl(args.features)
    ground_truth = load_ground_truth(args.ground_truth)
    frame = features.merge(
        ground_truth,
        on=KEY,
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(features) or len(frame) != len(ground_truth):
        raise ValueError(
            "feature and ground-truth query keys do not match exactly"
        )

    frame["fold"] = assign_folds(
        frame,
        fold_count=args.fold_count,
        fold_seed=args.fold_seed,
    )
    frame["tree_log"] = frame["candidate_tree_log_count"]
    frame["target_log"] = np.log(
        frame["exact_count"].to_numpy(dtype=np.float64)
    )
    columns = [
        "dataset",
        "query_size",
        "query_name",
        "fold",
        "tree_log",
        "target_log",
        "exact_count",
        *FEATURES,
    ]
    frame = frame[columns].sort_values(KEY).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(
        f"rows={len(frame)} datasets={frame['dataset'].nunique()} "
        f"query_sets={frame.groupby(['dataset', 'query_size']).ngroups} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
