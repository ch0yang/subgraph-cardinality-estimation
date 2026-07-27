# Subgraph Cardinality Estimation

Code and supplemental material for the accompanying subgraph cardinality
estimation paper.

## Requirements

- Python 3.10 or later
- CMake and a C++11 compiler

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

## Build

Build the feature exporter from the repository root:

```bash
cmake -S third_party/GQLBatchExport \
      -B third_party/GQLBatchExport/build
cmake --build third_party/GQLBatchExport/build --config Release -j
```

## Yeast Workload

The repository includes the complete yeast workload: one data graph and 1,800
query graphs generated with the SubgraphMatching benchmark query-generation
procedure used in the paper. `groundtruth/yeast.txt` contains exact counts for
the 1,567 positive-count queries used for training and evaluation.

Export the yeast features:

```bash
python scripts/export_features.py \
  --data-graph dataset/yeast/data_graph/yeast.graph \
  --ground-truth groundtruth/yeast.txt \
  --out-jsonl results/yeast_features.jsonl
```

Join the features with the positive exact counts:

```bash
python scripts/build_training_frame.py \
  --features results/yeast_features.jsonl \
  --ground-truth groundtruth/yeast.txt \
  --output results/yeast_training_frame.csv
```

Train and evaluate the estimator:

```bash
python src/candidate_tree_estimator/train.py \
  --frame results/yeast_training_frame.csv \
  --out-dir results/yeast_run
```

## Benchmark Data

The yeast data graph is distributed with the
[SubgraphMatching project](https://github.com/RapidsAtHKUST/SubgraphMatching).
The query graphs included here were generated with the benchmark
query-generation procedure used in the paper.

For another dataset, place its data graph and generated query workload under:

```text
dataset/<dataset>/data_graph/<dataset>.graph
dataset/<dataset>/query_graph/query_<dense|sparse>_<size>_<id>.graph
```

The compact ground-truth format is:

```text
query_name: exact_count
```

Use `scripts/export_features.py --help` and
`src/candidate_tree_estimator/train.py --help` for all options.

## Files

```text
src/candidate_tree_estimator/   Feature loading, model, and training code
scripts/                        Feature export and data preparation scripts
third_party/GQLBatchExport/     Candidate-space feature exporter
dataset/yeast/                  Complete yeast data and generated queries
groundtruth/yeast.txt           Positive exact counts used for training
supplementary/                  Supplemental PDF
```

`third_party/GQLBatchExport` is derived from SubgraphMatching and retains its
MIT license. See `THIRD_PARTY_NOTICES.md`.
