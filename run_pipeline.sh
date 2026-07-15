#!/usr/bin/env bash
set -euo pipefail
export STORAGE_BACKEND="${STORAGE_BACKEND:-local}"
python MLOps/pipeline_orchestration.py
