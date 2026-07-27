from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


FEATURES = [
    "candidate_tree_log_count",
    "candidate_domain_size_dispersion_mean",
    "sum_support_concentration",
    "candidate_relation_cycle_path_log_gap_mean",
    "probe_edge_feasible_sample_neglog_survival",
    "label_injectivity_product_log_sum",
    "probe_injective_sample_neglog_survival",
    "probe_full_match_sample_neglog_survival",
    "probe_graph_estimate_tree_log_gap",
    "probe_graph_estimate_signed_used_sample_fraction",
    "positive_sample_fraction_weighted_graph_gap_ratio",
]

GROUPS = [
    [0, 1, 2],
    [3, 4],
    [5, 6],
    [7, 8, 9, 10],
]

KEY = ["dataset", "query_size", "query_name"]
GROUND_TRUTH_COLUMNS = (
    "dataset",
    "query_size",
    "query_name",
    "exact_count",
)
QUERY_FILENAME_PATTERN = re.compile(
    r"query_(?:dense|sparse)_(\d+)_\d+\.graph",
    flags=re.IGNORECASE,
)


def query_key_from_path(value: str | Path) -> tuple[str, int, str]:
    normalized = str(value).replace("\\", "/")
    parts = Path(normalized).parts
    try:
        dataset_index = parts.index("dataset")
    except ValueError as error:
        raise ValueError(f"dataset component missing from {value}") from error
    directory_match = re.search(r"/query_(\d+)/", f"/{normalized}/")
    filename_match = QUERY_FILENAME_PATTERN.fullmatch(parts[-1])
    match = directory_match or filename_match
    if match is None:
        raise ValueError(f"query size missing from {value}")
    return (
        parts[dataset_index + 1].lower(),
        int(match.group(1)),
        parts[-1],
    )


def load_text_ground_truth(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    dataset = source.stem.lower()
    rows = []
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        query_name, separator, count_text = line.partition(":")
        match = QUERY_FILENAME_PATTERN.fullmatch(query_name.strip())
        if not separator or match is None:
            raise ValueError(
                f"invalid ground-truth row at {source}:{line_number}"
            )
        try:
            exact_count = int(count_text.strip())
        except ValueError as error:
            raise ValueError(
                f"invalid count at {source}:{line_number}"
            ) from error
        rows.append(
            {
                "dataset": dataset,
                "query_size": int(match.group(1)),
                "query_name": query_name.strip(),
                "exact_count": exact_count,
            }
        )
    return pd.DataFrame(rows, columns=GROUND_TRUTH_COLUMNS)


def load_ground_truth(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".txt":
        frame = load_text_ground_truth(source)
    else:
        frame = pd.read_csv(source)
    missing = sorted(set(GROUND_TRUTH_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"ground truth lacks columns: {missing}")
    frame = frame[list(GROUND_TRUTH_COLUMNS)].copy()
    frame["dataset"] = frame["dataset"].astype(str).str.lower()
    frame["query_size"] = frame["query_size"].astype(int)
    frame["query_name"] = frame["query_name"].astype(str)
    frame["exact_count"] = pd.to_numeric(
        frame["exact_count"],
        errors="raise",
    )
    if (frame["exact_count"] <= 0).any():
        raise ValueError("training ground truth must contain positive counts")
    if frame.duplicated(KEY).any():
        raise ValueError("duplicate query keys in ground truth")
    return frame


def query_paths_from_ground_truth(
    path: str | Path,
    root: str | Path,
) -> list[str]:
    frame = load_ground_truth(path)
    datasets = frame["dataset"].unique()
    if len(datasets) != 1:
        raise ValueError("feature export expects one dataset per ground-truth file")
    repository_root = Path(root)
    query_paths = []
    for row in frame.itertuples(index=False):
        query_path = (
            repository_root
            / "dataset"
            / row.dataset
            / "query_graph"
            / row.query_name
        )
        if not query_path.is_file():
            raise FileNotFoundError(f"query graph not found: {query_path}")
        query_paths.append(str(query_path.relative_to(repository_root)))
    return query_paths


def load_export_jsonl(path: str | Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            dataset, query_size, query_name = query_key_from_path(
                record["query_path"]
            )
            values = record.get("features", {}) or {}
            missing = sorted(set(FEATURES) - set(values))
            if missing:
                raise ValueError(f"missing features: {missing}")
            rows.append(
                {
                    "dataset": dataset,
                    "query_size": query_size,
                    "query_name": query_name,
                    **{name: float(values[name]) for name in FEATURES},
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError(f"no feature rows in {path}")
    if frame.duplicated(KEY).any():
        raise ValueError("duplicate query keys in exported features")
    return frame


def merge_exported_features(
    base: pd.DataFrame,
    exported: pd.DataFrame,
) -> pd.DataFrame:
    required = set(KEY)
    if not required.issubset(base.columns):
        raise ValueError(f"base frame lacks keys: {sorted(required - set(base))}")
    if not required.issubset(exported.columns):
        raise ValueError(
            f"export frame lacks keys: {sorted(required - set(exported))}"
        )
    drop = [name for name in FEATURES if name in base.columns]
    merged = base.drop(columns=drop).merge(
        exported,
        on=KEY,
        how="left",
        validate="one_to_one",
    )
    if merged[FEATURES].isna().any().any():
        raise ValueError("feature coverage is incomplete")
    return merged
