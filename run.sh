#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_executable="${PYTHON_BIN:-python3}"
exec "$python_executable" "$project_dir/analyze.py" --out "${1:-$project_dir/docs}"
