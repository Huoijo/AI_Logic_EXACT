#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_kaggle.sh smoke   # tiny unit smoke test, no LLM load
#   ./run_kaggle.sh batch          # dataset batch with FOL if present
#   ./run_kaggle.sh batch_nl       # force NL-premises -> logic -> answer
#   ./run_kaggle.sh benchmark      # same as batch, but writes richer metrics
# Optional env: INPUT_MODE=nl BATCH_SIZE=16 LIMIT=100 ./run_kaggle.sh benchmark

TASK="${1:-batch}"
INPUT_MODE="${INPUT_MODE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-0}"
LIMIT="${LIMIT:-}"
KERNEL="huoijo/exact-kaggle-core-xai"
BUILD_DIR=".kaggle_build"
OUT_DIR="kaggle_outputs"
ART_DIR="artifacts"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI not found. Install with: pip install kaggle"
  exit 1
fi

cleanup() {
  rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

echo "[1/5] Commit and push current repo"
git add .
git commit -m "kaggle-core ${TASK}" || true
git push

echo "[2/5] Build temporary Kaggle job folder"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cp kaggle_job/runner.py "$BUILD_DIR/runner.py"
cp kaggle_job/kernel-metadata.json "$BUILD_DIR/kernel-metadata.json"

TASK_FROM_SHELL="$TASK" INPUT_MODE="$INPUT_MODE" BATCH_SIZE="$BATCH_SIZE" LIMIT="$LIMIT" python - <<'PY_PATCH'
from pathlib import Path
import os, re
p = Path('.kaggle_build/runner.py')
s = p.read_text()
task = os.environ['TASK_FROM_SHELL']
s = re.sub(r'TASK = os\.environ\.get\("TASK", "[^"]+"\)', f'TASK = os.environ.get("TASK", "{task}")', s)
# Bake in optional defaults for Kaggle runner. Env can still override inside Kaggle.
for key in ["INPUT_MODE", "BATCH_SIZE", "LIMIT"]:
    val = os.environ.get(key)
    if val is not None:
        pattern = rf'{key} = os\.environ\.get\("{key}", "[^"]*"\)'
        s = re.sub(pattern, f'{key} = os.environ.get("{key}", "{val}")', s)
p.write_text(s)
PY_PATCH

echo "[3/5] Push Kaggle kernel"
TASK_FROM_SHELL="$TASK" INPUT_MODE="$INPUT_MODE" BATCH_SIZE="$BATCH_SIZE" LIMIT="$LIMIT" kaggle kernels push -p "$BUILD_DIR" --accelerator NvidiaTeslaT4

echo "[4/5] Wait for Kaggle artifact"

mkdir -p kaggle_outputs artifacts

ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT + 1))

  echo "Trying to download exact_artifacts.zip... attempt $ATTEMPT"

  rm -f kaggle_outputs/exact_artifacts.zip

  if kaggle kernels output "$KERNEL" \
      -p kaggle_outputs \
      -o \
      --file-pattern '^exact_artifacts\.zip$'; then

    if [ -f "kaggle_outputs/exact_artifacts.zip" ]; then
      echo "[ok] exact_artifacts.zip downloaded."
      break
    fi
  fi

  if [ "$ATTEMPT" -ge 60 ]; then
    echo "[error] Could not download exact_artifacts.zip after 60 attempts."
    echo "Open Kaggle web page to check whether the kernel failed or output file was not produced."
    exit 1
  fi

  sleep 20
done

echo "[5/5] Extract artifacts"
rm -rf artifacts
mkdir -p artifacts
unzip -o kaggle_outputs/exact_artifacts.zip -d artifacts
echo "Done. Check artifacts/"
