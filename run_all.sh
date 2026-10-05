#!/usr/bin/env bash
# Rebuild every output from the raw CFPB export.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" src/analyze.py
"$PYTHON_BIN" src/forecast_volume.py
"$PYTHON_BIN" src/model_relief.py
"$PYTHON_BIN" src/build_model_dashboard.py
"$PYTHON_BIN" src/write_model_report.py
"$PYTHON_BIN" src/verify_results.py
echo "Done. See analysis/, analysis/modeling/, figures/, report/."
