#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_kaggle.sh smoke
#   ./run_kaggle.sh batch
#   ./run_kaggle.sh batch_nl
#   ./run_kaggle.sh benchmark
#   ./run_kaggle.sh translate
#
# Optional env:
#   INPUT_MODE=nl BATCH_SIZE=16 LIMIT=100 ./run_kaggle.sh benchmark
#   RUN_ID=my_debug_run ./run_kaggle.sh batch_nl
#   MAX_ATTEMPTS=90 SLEEP_SECONDS=20 ./run_kaggle.sh batch_nl

TASK="${1:-batch}"
INPUT_MODE="${INPUT_MODE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-0}"
LIMIT="${LIMIT:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"

KERNEL="huoijo/exact-kaggle-core-xai"
BUILD_DIR=".kaggle_build"
OUT_DIR="kaggle_outputs"
ART_DIR="artifacts"
ARTIFACT_ZIP="exact_artifacts.zip"

MAX_ATTEMPTS="${MAX_ATTEMPTS:-90}"
SLEEP_SECONDS="${SLEEP_SECONDS:-20}"

START_TS="$(date +%s)"

format_duration() {
  local total="$1"
  local h=$((total / 3600))
  local m=$(((total % 3600) / 60))
  local s=$((total % 60))

  if [ "$h" -gt 0 ]; then
    printf "%02dh:%02dm:%02ds" "$h" "$m" "$s"
  else
    printf "%02dm:%02ds" "$m" "$s"
  fi
}

elapsed() {
  local now
  now="$(date +%s)"
  format_duration $((now - START_TS))
}

notify_mac() {
  local title="$1"
  local message="$2"

  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display notification \"$message\" with title \"$title\"" >/dev/null 2>&1 || true
  fi
}

finish_success() {
  local e
  e="$(elapsed)"
  echo
  echo "✅ Done in $e. Check ${ART_DIR}/"
  notify_mac "EXACT Kaggle finished" "Task ${TASK} completed in ${e}."
  printf '\a' || true
}

finish_fail() {
  local e
  e="$(elapsed)"
  echo
  echo "❌ Failed after $e."
  notify_mac "EXACT Kaggle failed" "Task ${TASK} failed after ${e}."
  printf '\a' || true
}

trap finish_fail ERR

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI not found. Install with: pip install kaggle"
  exit 1
fi

cleanup() {
  rm -rf "$BUILD_DIR"
}
trap cleanup EXIT

echo "======================================"
echo "EXACT Kaggle Core Runner"
echo "TASK       = $TASK"
echo "INPUT_MODE = $INPUT_MODE"
echo "BATCH_SIZE = $BATCH_SIZE"
echo "LIMIT      = ${LIMIT:-<none>}"
echo "RUN_ID     = $RUN_ID"
echo "KERNEL     = $KERNEL"
echo "START      = $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================"

echo "[$(elapsed)] [0/5] Ensure generated files are ignored"
touch .gitignore

grep -qxF "artifacts/" .gitignore || echo "artifacts/" >> .gitignore
grep -qxF "kaggle_outputs/" .gitignore || echo "kaggle_outputs/" >> .gitignore
grep -qxF ".kaggle_build/" .gitignore || echo ".kaggle_build/" >> .gitignore
grep -qxF "*.zip" .gitignore || echo "*.zip" >> .gitignore

git rm -r --cached artifacts kaggle_outputs .kaggle_build 2>/dev/null || true

echo "[$(elapsed)] [1/5] Commit and push current repo"
git add .
git commit -m "kaggle-core ${TASK}" || true
git push

echo "[$(elapsed)] [2/5] Build temporary Kaggle job folder"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

cp kaggle_job/runner.py "$BUILD_DIR/runner.py"
cp kaggle_job/kernel-metadata.json "$BUILD_DIR/kernel-metadata.json"

TASK_FROM_SHELL="$TASK" \
INPUT_MODE="$INPUT_MODE" \
BATCH_SIZE="$BATCH_SIZE" \
LIMIT="$LIMIT" \
RUN_ID="$RUN_ID" \
MODEL_NAME="$MODEL_NAME" \
python - <<'PY_PATCH'
from pathlib import Path
import os
import re

p = Path(".kaggle_build/runner.py")
s = p.read_text()

replacements = {
    "TASK": os.environ.get("TASK_FROM_SHELL", "batch"),
    "INPUT_MODE": os.environ.get("INPUT_MODE", "auto"),
    "BATCH_SIZE": os.environ.get("BATCH_SIZE", "0"),
    "LIMIT": os.environ.get("LIMIT", ""),
    "RUN_ID": os.environ.get("RUN_ID", ""),
    "MODEL_NAME": os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B"),
}

for key, val in replacements.items():
    pattern = rf'{key}\s*=\s*os\.environ\.get\("{key}",\s*"[^"]*"\)'
    replacement = f'{key} = os.environ.get("{key}", "{val}")'

    if re.search(pattern, s):
        s = re.sub(pattern, replacement, s)
    else:
        if key == "RUN_ID":
            insert_after = "from pathlib import Path\n"
            if insert_after in s:
                s = s.replace(insert_after, insert_after + replacement + "\n", 1)
            else:
                s = replacement + "\n" + s

p.write_text(s)
PY_PATCH

echo "[$(elapsed)] [3/5] Push Kaggle kernel"
kaggle kernels push -p "$BUILD_DIR" --accelerator NvidiaTeslaT4

echo "[$(elapsed)] [4/5] Wait for Kaggle artifact"
mkdir -p "$OUT_DIR" "$ART_DIR"

ATTEMPT=0

while true; do
  ATTEMPT=$((ATTEMPT + 1))

  echo "[$(elapsed)] Trying to download ${ARTIFACT_ZIP}... attempt ${ATTEMPT}/${MAX_ATTEMPTS}"

  rm -f "${OUT_DIR}/${ARTIFACT_ZIP}"

  if kaggle kernels output "$KERNEL" \
      -p "$OUT_DIR" \
      -o \
      --file-pattern "^${ARTIFACT_ZIP}$"; then

    if [ -f "${OUT_DIR}/${ARTIFACT_ZIP}" ]; then
      echo "[$(elapsed)] [ok] ${ARTIFACT_ZIP} downloaded."
      break
    fi
  fi

  if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
    echo "[$(elapsed)] [error] Could not download ${ARTIFACT_ZIP} after ${MAX_ATTEMPTS} attempts."
    echo "Open Kaggle web page to check whether the kernel failed or output file was not produced:"
    echo "https://www.kaggle.com/code/${KERNEL}"
    exit 1
  fi

  echo "[$(elapsed)] Artifact not ready yet. Sleeping ${SLEEP_SECONDS}s..."
  sleep "$SLEEP_SECONDS"
done

echo "[$(elapsed)] [5/5] Extract artifacts"
rm -rf "$ART_DIR"
mkdir -p "$ART_DIR"
unzip -o "${OUT_DIR}/${ARTIFACT_ZIP}" -d "$ART_DIR"

if [ -f "${ART_DIR}/run_id.txt" ]; then
  GOT_RUN_ID="$(cat "${ART_DIR}/run_id.txt" | tr -d '\n\r')"
  if [ "$GOT_RUN_ID" != "$RUN_ID" ]; then
    echo "[error] Downloaded artifact run_id=${GOT_RUN_ID}, expected ${RUN_ID}."
    echo "This looks like a stale artifact from a previous Kaggle run."
    exit 1
  fi
  echo "[$(elapsed)] [ok] run_id verified: ${GOT_RUN_ID}"
else
  echo "[$(elapsed)] [warn] artifacts/run_id.txt not found. Cannot verify whether artifact is stale."
fi

if [ -f "${ART_DIR}/eval_report.json" ]; then
  echo "----- eval_report.json -----"
  cat "${ART_DIR}/eval_report.json"
  echo
fi

trap - ERR
finish_success