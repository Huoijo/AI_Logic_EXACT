from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# These defaults can be overridden from run_kaggle.sh via env/patching.
REPO_URL = os.environ.get("REPO_URL", "https://github.com/Huoijo/AI_Logic_EXACT.git")
BRANCH = os.environ.get("BRANCH", "main")
TASK = os.environ.get("TASK", "batch")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
USE_LLM = os.environ.get("USE_LLM", "1")
LOAD_4BIT = os.environ.get("LOAD_4BIT", "1")
INPUT_MODE = os.environ.get("INPUT_MODE", "auto")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "0")
LIMIT = os.environ.get("LIMIT", "")
DATASET = os.environ.get("DATASET", "data/fraction_dataset.json")
RUN_ID = os.environ.get("RUN_ID", "")
LOG_EACH_CASE = os.environ.get("LOG_EACH_CASE", "1")

# Important: do NOT clone repo into /kaggle/working.
# Kaggle persists /kaggle/working as downloadable output, so putting the repo there
# causes the whole source tree to be downloaded by `kaggle kernels output`.
RUNTIME_DIR = Path("/tmp/exact_kaggle_runtime")
REPO_DIR = RUNTIME_DIR / "AI_Logic_EXACT"
OUT_DIR = RUNTIME_DIR / "outputs"
FINAL_ZIP = Path("/kaggle/working/exact_artifacts.zip")


def run(cmd, cwd=None, env=None):
    print("RUN:", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, cwd=cwd, env=env)


print("=" * 60, flush=True)
print("EXACT Kaggle Runtime", flush=True)
print(f"TASK={TASK}", flush=True)
print(f"MODEL_NAME={MODEL_NAME}", flush=True)
print(f"INPUT_MODE={INPUT_MODE}", flush=True)
print(f"BATCH_SIZE={BATCH_SIZE}", flush=True)
print(f"LIMIT={LIMIT or '<none>'}", flush=True)
print(f"DATASET={DATASET}", flush=True)
print(f"RUN_ID={RUN_ID}", flush=True)
print(f"LOG_EACH_CASE={LOG_EACH_CASE}", flush=True)
print("=" * 60, flush=True)

if RUNTIME_DIR.exists():
    shutil.rmtree(RUNTIME_DIR)
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, REPO_DIR])
run(["pip", "install", "-q", "-r", REPO_DIR / "requirements_kaggle.txt"])

child_env = os.environ.copy()
child_env["MODEL_NAME"] = MODEL_NAME
child_env["USE_LLM"] = USE_LLM
child_env["LOAD_4BIT"] = LOAD_4BIT
child_env["PYTHONUNBUFFERED"] = "1"

# Resolve dataset relative to repo unless an absolute path is provided.
dataset_path = Path(DATASET)
if not dataset_path.is_absolute():
    dataset_path = REPO_DIR / dataset_path

runner_args = [
    "python", "-u", REPO_DIR / "scripts" / "kaggle_core_runner.py",
    "--task", TASK,
    "--dataset", dataset_path,
    "--out", OUT_DIR,
    "--input-mode", INPUT_MODE,
    "--batch-size", BATCH_SIZE,
]
if LIMIT:
    runner_args.extend(["--limit", LIMIT])
if TASK == "batch_nl" or TASK == "translate":
    runner_args[runner_args.index("--input-mode") + 1] = "nl"
if TASK == "smoke":
    # smoke now runs a tiny unit chain and does NOT load the LLM
    child_env["USE_LLM"] = "0"
if LOG_EACH_CASE == "1" and TASK != "smoke":
    runner_args.append("--log-cases")

run(runner_args, env=child_env)

# Persist run marker so the local downloader can detect stale artifact zips.
(OUT_DIR / "run_id.txt").write_text(RUN_ID, encoding="utf-8")
(OUT_DIR / "runtime_config.json").write_text(
    __import__("json").dumps({
        "task": TASK,
        "model_name": MODEL_NAME,
        "input_mode": INPUT_MODE,
        "batch_size": BATCH_SIZE,
        "limit": LIMIT,
        "dataset": DATASET,
        "run_id": RUN_ID,
        "log_each_case": LOG_EACH_CASE,
    }, indent=2),
    encoding="utf-8",
)

# Only persist one file in /kaggle/working.
# Everything else stays in /tmp and will not be downloaded as kernel output.
if FINAL_ZIP.exists():
    FINAL_ZIP.unlink()
run(["bash", "-lc", f"cd {OUT_DIR} && zip -r {FINAL_ZIP} ."])
print(f"DONE: {FINAL_ZIP}", flush=True)
