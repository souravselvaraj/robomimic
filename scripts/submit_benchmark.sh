#!/bin/bash
# Submit benchmark training jobs. Optionally filter by substring:
#   ./scripts/submit_benchmark.sh              # submit all 30
#   ./scripts/submit_benchmark.sh fm_          # flow matching only
#   ./scripts/submit_benchmark.sh dp_square    # DP on square, all seeds
set -eu
cd "$(dirname "$0")/.."

FILTER="${1:-}"
for config in configs/benchmark/*.json; do
  name=$(basename "$config" .json)
  if [[ -n "$FILTER" && "$name" != *"$FILTER"* ]]; then
    continue
  fi
  sbatch -J "$name" scripts/train_benchmark.sbatch "$config"
done
