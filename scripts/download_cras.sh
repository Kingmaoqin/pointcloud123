#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p data/raw outputs/logs

echo "hostname=$(hostname)" | tee outputs/logs/download_cras.log
df -h . | tee -a outputs/logs/download_cras.log

curl -L -C - -o data/raw/craslabbim.ifc \
  "https://zenodo.org/records/7948116/files/craslabbim.ifc?download=1"
curl -L -C - -o data/raw/craslabannotated.zip \
  "https://zenodo.org/records/7948116/files/craslabannotated.zip?download=1"

python - <<'PY'
from patent_gap.data.audit import file_md5
expected = {
    "data/raw/craslabannotated.zip": "e5ecedab8f2a1d1f91861a3aec028a72",
    "data/raw/craslabbim.ifc": "e20658f0d2d9e13c62363169b7fa3193",
}
for path, md5 in expected.items():
    actual = file_md5(path)
    print(f"{path}: {actual} expected={md5} ok={actual == md5}")
PY

