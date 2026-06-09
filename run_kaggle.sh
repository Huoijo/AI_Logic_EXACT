#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-batch}"
INPUT_MODE="${INPUT_MODE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-0}"
LIMIT="${LIMIT:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
DATASET="${DATASET:-data/fraction_dataset.json}"
LOG_EACH_CASE="${LOG_EACH_CASE:-1}"
REPO_URL="${REPO_URL:-https://github.com/Huoijo/AI_Logic_EXACT.git}"
KERNEL="${KERNEL:-huoijo/exact-kaggle-core-xai}"

BUILD_DIR=".kaggle_build"
OUT_DIR="kaggle_outputs"
ART_DIR="artifacts"
ARTIFACT_ZIP="exact_artifacts.zip"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-900}"
SLEEP_SECONDS="${SLEEP_SECONDS:-20}"
START_TS="$(date +%s)"

format_duration() { local total="$1"; local h=$((total/3600)); local m=$(((total%3600)/60)); local s=$((total%60)); if [ "$h" -gt 0 ]; then printf "%02dh:%02dm:%02ds" "$h" "$m" "$s"; else printf "%02dm:%02ds" "$m" "$s"; fi; }
elapsed() { local now; now="$(date +%s)"; format_duration $((now-START_TS)); }
notify_mac() { local title="$1"; local message="$2"; if command -v osascript >/dev/null 2>&1; then osascript -e "display notification \"$message\" with title \"$title\"" >/dev/null 2>&1 || true; fi; }
finish_fail() { local e; e="$(elapsed)"; echo; echo "❌ Failed after $e."; notify_mac "EXACT Kaggle failed" "Task ${TASK} failed after ${e}."; printf '\a' || true; }
finish_success() { local e; e="$(elapsed)"; echo; echo "✅ Done in $e. Check ${ART_DIR}/"; notify_mac "EXACT Kaggle finished" "Task ${TASK} completed in ${e}."; printf '\a' || true; }
trap finish_fail ERR

if ! command -v kaggle >/dev/null 2>&1; then echo "Kaggle CLI not found. Install with: pip install kaggle"; exit 1; fi
cleanup() { rm -rf "$BUILD_DIR"; }
trap cleanup EXIT
KAGGLE_URL="https://www.kaggle.com/code/${KERNEL}"

echo "======================================"
echo "EXACT Kaggle Core Runner"
echo "TASK          = $TASK"
echo "INPUT_MODE    = $INPUT_MODE"
echo "MODEL_NAME    = $MODEL_NAME"
echo "ADAPTER_PATH  = ${ADAPTER_PATH:-<none>}"
echo "DATASET       = $DATASET"
echo "BATCH_SIZE    = $BATCH_SIZE"
echo "LIMIT         = ${LIMIT:-<none>}"
echo "LOG_EACH_CASE = $LOG_EACH_CASE"
echo "RUN_ID        = $RUN_ID"
echo "KAGGLE URL    = $KAGGLE_URL"
echo "START         = $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================"

echo "[$(elapsed)] [0/5] Ensure generated files are ignored"
touch .gitignore
for pat in "artifacts/" "kaggle_outputs/" ".kaggle_build/" "*.zip"; do grep -qxF "$pat" .gitignore || echo "$pat" >> .gitignore; done
git rm -r --cached artifacts kaggle_outputs .kaggle_build 2>/dev/null || true

echo "[$(elapsed)] [1/5] Commit and push current repo"
git add .
git commit -m "kaggle-core ${TASK}" || true
git push

echo "[$(elapsed)] [2/5] Build temporary Kaggle job folder"
rm -rf "$BUILD_DIR"; mkdir -p "$BUILD_DIR"
cp kaggle_job/runner.py "$BUILD_DIR/runner.py"
cp kaggle_job/kernel-metadata.json "$BUILD_DIR/kernel-metadata.json"

TASK_FROM_SHELL="$TASK" INPUT_MODE="$INPUT_MODE" BATCH_SIZE="$BATCH_SIZE" LIMIT="$LIMIT" RUN_ID="$RUN_ID" \
MODEL_NAME="$MODEL_NAME" ADAPTER_PATH="$ADAPTER_PATH" DATASET="$DATASET" LOG_EACH_CASE="$LOG_EACH_CASE" REPO_URL="$REPO_URL" KERNEL="$KERNEL" \
python - <<'PY_PATCH'
from pathlib import Path
import json, os, re
runner = Path('.kaggle_build/runner.py')
s = runner.read_text()
replacements = {
  'REPO_URL': os.environ.get('REPO_URL',''),
  'TASK': os.environ.get('TASK_FROM_SHELL','batch'),
  'INPUT_MODE': os.environ.get('INPUT_MODE','auto'),
  'BATCH_SIZE': os.environ.get('BATCH_SIZE','0'),
  'LIMIT': os.environ.get('LIMIT',''),
  'RUN_ID': os.environ.get('RUN_ID',''),
  'MODEL_NAME': os.environ.get('MODEL_NAME','Qwen/Qwen3-8B'),
  'ADAPTER_PATH': os.environ.get('ADAPTER_PATH',''),
  'DATASET': os.environ.get('DATASET','data/fraction_dataset.json'),
  'LOG_EACH_CASE': os.environ.get('LOG_EACH_CASE','1'),
}
for key, val in replacements.items():
    pat = rf'{key}\s*=\s*os\.environ\.get\("{key}",\s*"[^"]*"\)'
    repl = f'{key} = os.environ.get("{key}", "{val}")'
    s = re.sub(pat, repl, s)
runner.write_text(s)
meta_path = Path('.kaggle_build/kernel-metadata.json')
meta = json.loads(meta_path.read_text())
meta['id'] = os.environ.get('KERNEL', meta.get('id',''))
meta['code_file'] = 'runner.py'
meta.setdefault('is_private','false')
meta_path.write_text(json.dumps(meta, indent=2))
PY_PATCH

echo "[$(elapsed)] [3/5] Push Kaggle kernel"
PUSH_LOG="$(mktemp)"
if ! kaggle kernels push -p "$BUILD_DIR" --accelerator NvidiaTeslaT4 2>&1 | tee "$PUSH_LOG"; then
  echo "[$(elapsed)] [error] Kaggle kernel push failed. Not waiting for artifact."; cat "$PUSH_LOG"; exit 1
fi
if grep -Eiq "Kernel push error|Maximum .* session count|Permission .* denied|error:" "$PUSH_LOG"; then
  echo "[$(elapsed)] [error] Kaggle reported a push error. Not waiting for artifact."; cat "$PUSH_LOG"; exit 1
fi

echo "[$(elapsed)] [4/5] Wait for Kaggle artifact"
mkdir -p "$OUT_DIR" "$ART_DIR"
ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT+1))
  echo "[$(elapsed)] Trying to download ${ARTIFACT_ZIP}... attempt ${ATTEMPT}/${MAX_ATTEMPTS}"
  rm -rf "${OUT_DIR:?}/"*; mkdir -p "$OUT_DIR"
  kaggle kernels output "$KERNEL" -p "$OUT_DIR" -o --file-pattern "^${ARTIFACT_ZIP}$" || true
  if [ -f "${OUT_DIR}/${ARTIFACT_ZIP}" ]; then echo "[$(elapsed)] [ok] ${ARTIFACT_ZIP} downloaded."; break; fi
  if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then echo "[$(elapsed)] [error] Could not download artifact. Check $KAGGLE_URL"; exit 1; fi
  echo "[$(elapsed)] Artifact not ready yet. Sleeping ${SLEEP_SECONDS}s..."; sleep "$SLEEP_SECONDS"
done

echo "[$(elapsed)] [5/5] Extract artifacts"
rm -rf "$ART_DIR"; mkdir -p "$ART_DIR"
unzip -o "${OUT_DIR}/${ARTIFACT_ZIP}" -d "$ART_DIR"
if [ -f "${ART_DIR}/run_id.txt" ]; then GOT_RUN_ID="$(cat "${ART_DIR}/run_id.txt" | tr -d '\n\r')"; if [ "$GOT_RUN_ID" = "$RUN_ID" ]; then echo "[$(elapsed)] [ok] run_id verified: ${GOT_RUN_ID}"; else echo "[warn] run_id=${GOT_RUN_ID}, expected=${RUN_ID}"; fi; fi
if [ -f "${ART_DIR}/eval_report.json" ]; then echo "----- eval_report.json -----"; cat "${ART_DIR}/eval_report.json"; echo; fi
if [ -f "${ART_DIR}/qa_report.md" ]; then echo "[$(elapsed)] QA report ready: ${ART_DIR}/qa_report.md"; fi
trap - ERR
finish_success
