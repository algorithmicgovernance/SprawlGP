#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--apply" ]]; then
  cat <<'EOF'
DRY RUN ONLY

This cleanup keeps the selected baseline and retained V1 experiment, moves
existing V1 outputs into experiments/v1, and removes obsolete V2/V3 Logistic
Regression code/configs/outputs.

Before applying, first regenerate the selected baseline and verify that its
summary matches the old selected V3 result.

Apply with:
  bash scripts/cleanup_logistic_modeling.sh --apply
EOF
  exit 0
fi

mkdir -p \
  artifacts/models/logistic_regression/experiments/v1 \
  reports/modeling/logistic_regression/experiments/v1 \
  data/metadata/modeling/logistic_regression/experiments/v1

if [[ -d artifacts/models/logistic_v1 ]]; then
  cp -R artifacts/models/logistic_v1/. \
    artifacts/models/logistic_regression/experiments/v1/
  rm -rf artifacts/models/logistic_v1
fi

if [[ -d reports/modeling_v1 ]]; then
  cp -R reports/modeling_v1/. \
    reports/modeling/logistic_regression/experiments/v1/
  rm -rf reports/modeling_v1
fi

for file in feature_manifest.yaml split_manifest.csv; do
  if [[ -f "data/metadata/modeling/$file" ]]; then
    mv "data/metadata/modeling/$file" \
      "data/metadata/modeling/logistic_regression/experiments/v1/$file"
  fi
done

rm -f \
  configs/modeling/experiment_v1.yaml \
  configs/modeling/experiment_v2.yaml \
  configs/modeling/experiment_v3.yaml \
  src/models/train_logistic_rolling.py \
  src/models/data.py \
  tests/test_modeling.py \
  tests/test_logistic_rolling.py

rm -rf \
  artifacts/models/logistic_v2 \
  artifacts/models/logistic_v3 \
  data/metadata/modeling_v2 \
  data/metadata/modeling_v3 \
  reports/modeling_v2 \
  reports/modeling_v3 \
  artifacts/archive/logistic_v2_* \
  reports/archive/logistic_v2_*

find src tests -type d -name '__pycache__' -prune -exec rm -rf {} +

echo "Logistic Regression cleanup complete."
