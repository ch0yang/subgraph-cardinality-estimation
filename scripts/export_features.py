from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from candidate_tree_estimator.features import query_paths_from_ground_truth


def main() -> None:
    parser = argparse.ArgumentParser()
    query_source = parser.add_mutually_exclusive_group(required=True)
    query_source.add_argument("--query-list", type=Path)
    query_source.add_argument("--ground-truth", type=Path)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--data-graph", type=Path, required=True)
    parser.add_argument(
        "--binary",
        type=Path,
        default=None,
        help="Path to GQLBatchExport; inferred from the build directory by default.",
    )
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--tree-seed", type=int, default=42)
    parser.add_argument("--graph-seed", type=int, default=43)
    parser.add_argument(
        "--candidate-tree-strategy",
        choices=(
            "current_bfs_maxcand",
            "bfs_mincand_root",
            "bfs_max_query_degree_root",
            "bfs_min_query_degree_root",
            "mwst_low_density_edges",
            "mwst_low_count_edges",
            "mwst_high_density_edges",
        ),
        default="mwst_low_density_edges",
    )
    parser.add_argument("--gql-refinement-rounds", type=int, default=5)
    parser.add_argument("--cycle-path-budget", type=int, default=5_000_000)
    args = parser.parse_args()

    build_dir = (
        ROOT
        / "third_party"
        / "GQLBatchExport"
        / "build"
        / "matching"
    )
    candidates = [
        args.binary,
        build_dir / "GQLBatchExport.out",
        build_dir / "GQLBatchExport.exe",
        build_dir / "Release" / "GQLBatchExport.exe",
    ]
    binary = next(
        (path for path in candidates if path is not None and path.is_file()),
        None,
    )
    if binary is None:
        searched = ", ".join(
            str(path) for path in candidates if path is not None
        )
        raise FileNotFoundError(f"GQLBatchExport not found; searched: {searched}")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        query_list = args.query_list
        if args.ground_truth is not None:
            temporary_dir = Path(
                stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="sce_queries_")
                )
            )
            query_list = temporary_dir / "queries.txt"
            query_paths = query_paths_from_ground_truth(
                args.ground_truth,
                ROOT,
            )
            query_list.write_text(
                "\n".join(query_paths) + "\n",
                encoding="utf-8",
            )
        command = [
            str(binary),
            "-d",
            str(args.data_graph),
            "--query_list",
            str(query_list),
            "--out_jsonl",
            str(args.out_jsonl),
            "--mode",
            "pruned",
            "--payload",
            "final11d_features",
            "--emit_timing",
            "--full_probe_budget",
            "0",
            "--full_probe_count_cap",
            "1000",
            "--tree_sample_count",
            str(args.budget),
            "--tree_sample_seed",
            str(args.tree_seed),
            "--tree_sample_strategy",
            "weighted",
            "--candidate_tree_strategy",
            args.candidate_tree_strategy,
            "--probe_fastest2_cap",
            str(args.budget),
            "--probe_fastest2_success_threshold",
            "0",
            "--probe_fastest2_seed",
            str(args.graph_seed),
            "--cycle_path_budget",
            str(args.cycle_path_budget),
            "--gql_refinement_rounds",
            str(args.gql_refinement_rounds),
        ]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
