#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_kaggle.sh smoke
#   ./run_kaggle.sh batch
#   ./run_kaggle.sh batch_nl
#   ./run_kaggle.sh benchmark
#   ./run_kaggle.sh translate
#
# Optional env examples:
#   INPUT_MODE=nl DATASET=data/full_dataset.json MODEL_NAME="Qwen/Qwen3-0.6B" LIMIT=50 BATCH_SIZE=16 ./run_kaggle.sh benchmark
#   LOG_EACH_CASE=0 ./run_kaggle.sh benchmark
#   KERNEL="yourname/exact-kaggle-core-xai" REPO_URL="https://github.com/you/repo.git" ./run_kaggle.sh batch_nl

TASK="${1:-batch}"
INPUT_MODE="${INPUT_MODE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-0}"
LIMIT="${LIMIT:-}"
CASE_IDS="${CASE_IDS:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-data/fraction_dataset.json}"
LOG_EACH_CASE="${LOG_EACH_CASE:-1}"
REPO_URL="${REPO_URL:-https://github.com/Huoijo/AI_Logic_EXACT.git}"
KERNEL="${KERNEL:-huoijo/exact-kaggle-core-xai}"

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

finish_fail() {
  local e
  e="$(elapsed)"
  echo
  echo "❌ Failed after $e."
  notify_mac "EXACT Kaggle failed" "Task ${TASK} failed after ${e}."
  printf '\a' || true
}

finish_success() {
  local e
  e="$(elapsed)"
  echo
  echo "✅ Done in $e. Check ${ART_DIR}/"
  notify_mac "EXACT Kaggle finished" "Task ${TASK} completed in ${e}."
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

KAGGLE_URL="https://www.kaggle.com/code/${KERNEL}"

echo "======================================"
echo "EXACT Kaggle Core Runner"
echo "TASK          = $TASK"
echo "INPUT_MODE    = $INPUT_MODE"
echo "MODEL_NAME    = $MODEL_NAME"
echo "DATASET       = $DATASET"
echo "BATCH_SIZE    = $BATCH_SIZE"
echo "LIMIT         = ${LIMIT:-<none>}"
echo "CASE_IDS      = ${CASE_IDS:-<none>}"
echo "LOG_EACH_CASE = $LOG_EACH_CASE"
echo "RUN_ID        = $RUN_ID"
echo "KERNEL        = $KERNEL"
echo "KAGGLE URL    = $KAGGLE_URL"
echo "START         = $(date '+%Y-%m-%d %H:%M:%S')"
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
DATASET="$DATASET" \
LOG_EACH_CASE="$LOG_EACH_CASE" \
REPO_URL="$REPO_URL" \
KERNEL="$KERNEL" \
python - <<'PY_PATCH'
from pathlib import Path
import json
import os
import re

runner = Path(".kaggle_build/runner.py")
s = runner.read_text()

replacements = {
    "REPO_URL": os.environ.get("REPO_URL", ""),
    "TASK": os.environ.get("TASK_FROM_SHELL", "batch"),
    "INPUT_MODE": os.environ.get("INPUT_MODE", "auto"),
    "BATCH_SIZE": os.environ.get("BATCH_SIZE", "0"),
    "LIMIT": os.environ.get("LIMIT", ""),
    "RUN_ID": os.environ.get("RUN_ID", ""),
    "MODEL_NAME": os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B"),
    "DATASET": os.environ.get("DATASET", "data/fraction_dataset.json"),
    "LOG_EACH_CASE": os.environ.get("LOG_EACH_CASE", "1"),
}

for key, val in replacements.items():
    # KEY = os.environ.get("KEY", "...")
    pattern_env = rf'{key}\s*=\s*os\.environ\.get\("{key}",\s*"[^"]*"\)'
    repl_env = f'{key} = os.environ.get("{key}", "{val}")'
    # KEY = "..."
    pattern_plain = rf'{key}\s*=\s*"[^"]*"'
    repl_plain = repl_env
    if re.search(pattern_env, s):
        s = re.sub(pattern_env, repl_env, s)
    elif re.search(pattern_plain, s):
        s = re.sub(pattern_plain, repl_plain, s)
    else:
        insert_after = "from pathlib import Path\n"
        if insert_after in s:
            s = s.replace(insert_after, insert_after + repl_env + "\n", 1)
        else:
            s = repl_env + "\n" + s

runner.write_text(s)

metadata_path = Path(".kaggle_build/kernel-metadata.json")
metadata = json.loads(metadata_path.read_text())
metadata["id"] = os.environ.get("KERNEL", metadata.get("id", ""))
# Public avoids some CLI output permission issues. Change to "true" if you explicitly want private.
metadata.setdefault("is_private", "false")
metadata_path.write_text(json.dumps(metadata, indent=2))
PY_PATCH

echo "[$(elapsed)] [3/5] Push Kaggle kernel"

PUSH_LOG="$(mktemp)"
if ! kaggle kernels push -p "$BUILD_DIR" --accelerator NvidiaTeslaT4 2>&1 | tee "$PUSH_LOG"; then
  echo "[$(elapsed)] [error] Kaggle kernel push failed. Not waiting for artifact."
  cat "$PUSH_LOG"
  exit 1
fi

if grep -Eiq "Kernel push error|Maximum .* session count|Permission .* denied|error:" "$PUSH_LOG"; then
  echo "[$(elapsed)] [error] Kaggle reported a push error. Not waiting for artifact."
  cat "$PUSH_LOG"
  exit 1
fi

echo "[$(elapsed)] [4/5] Wait for Kaggle artifact"
mkdir -p "$OUT_DIR" "$ART_DIR"

ATTEMPT=0
MAX_ATTEMPTS="${MAX_ATTEMPTS:-90}"
SLEEP_SECONDS="${SLEEP_SECONDS:-20}"

while true; do
  ATTEMPT=$((ATTEMPT + 1))
  echo "[$(elapsed)] Trying to download ${ARTIFACT_ZIP}... attempt ${ATTEMPT}/${MAX_ATTEMPTS}"

  rm -rf "${OUT_DIR:?}/"*
  mkdir -p "$OUT_DIR"

  # 1) Try the preferred zip artifact first
  kaggle kernels output "$KERNEL" \
    -p "$OUT_DIR" \
    -o \
    --file-pattern "^${ARTIFACT_ZIP}$" || true

  if [ -f "${OUT_DIR}/${ARTIFACT_ZIP}" ]; then
    echo "[$(elapsed)] [ok] ${ARTIFACT_ZIP} downloaded."
    DOWNLOAD_MODE="zip"
    break
  fi

  # 2) Fallback: try raw outputs folder
  echo "[$(elapsed)] Zip not found; trying raw outputs/* fallback..."
  kaggle kernels output "$KERNEL" \
    -p "$OUT_DIR" \
    -o \
    --file-pattern '^outputs/.*' || true

  if find "$OUT_DIR" -type f | grep -q .; then
    echo "[$(elapsed)] [ok] Raw outputs downloaded:"
    find "$OUT_DIR" -maxdepth 4 -type f
    DOWNLOAD_MODE="raw"
    break
  fi

  if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
    echo "[$(elapsed)] [error] Could not download artifact after ${MAX_ATTEMPTS} attempts."
    echo "Debug manually with:"
    echo "  kaggle kernels output $KERNEL -p kaggle_outputs/debug -o"
    echo "  find kaggle_outputs/debug -maxdepth 4 -type f"
    exit 1
  fi

  echo "[$(elapsed)] Artifact not ready yet. Sleeping ${SLEEP_SECONDS}s..."
  sleep "$SLEEP_SECONDS"
done

echo "[$(elapsed)] [5/5] Extract artifacts"
rm -rf "$ART_DIR"
mkdir -p "$ART_DIR"

if [ "$DOWNLOAD_MODE" = "zip" ]; then
  unzip -o "${OUT_DIR}/${ARTIFACT_ZIP}" -d "$ART_DIR"
else
  if [ -d "${OUT_DIR}/outputs" ]; then
    cp -R "${OUT_DIR}/outputs/." "$ART_DIR/"
  else
    cp -R "${OUT_DIR}/." "$ART_DIR/"
  fi
fi

if [ -f "${ART_DIR}/run_id.txt" ]; then
  GOT_RUN_ID="$(cat "${ART_DIR}/run_id.txt" | tr -d '\n\r')"
  if [ "$GOT_RUN_ID" != "$RUN_ID" ]; then
    echo "[warn] Downloaded artifact run_id=${GOT_RUN_ID}, expected ${RUN_ID}."
    echo "[warn] This may be a stale artifact from a previous Kaggle run."
  else
    echo "[$(elapsed)] [ok] run_id verified: ${GOT_RUN_ID}"
  fi
else
  echo "[$(elapsed)] [warn] artifacts/run_id.txt not found. Cannot verify run_id."
fi

if [ -f "${ART_DIR}/eval_report.json" ]; then
  echo "----- eval_report.json -----"
  cat "${ART_DIR}/eval_report.json"
  echo
fi

if [ -f "${ART_DIR}/qa_report.md" ]; then
  echo "[$(elapsed)] QA report ready: ${ART_DIR}/qa_report.md"
fi

trap - ERR
finish_success

  if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
    echo "[$(elapsed)] [error] Could not download ${ARTIFACT_ZIP} after ${MAX_ATTEMPTS} attempts."
    echo "Open Kaggle web page to check whether the kernel failed or output file was not produced:"
    echo "$KAGGLE_URL"
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

if [ -f "${ART_DIR}/qa_report.md" ]; then
  echo "[$(elapsed)] Readable report generated: ${ART_DIR}/qa_report.md"
fi

trap - ERR
finish_success
