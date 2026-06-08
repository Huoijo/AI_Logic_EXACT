from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# TODO: đổi URL repo của bạn ở đây
REPO_URL = "https://github.com/Huoijo/AI_Logic_EXACT"
BRANCH = os.environ.get("BRANCH", "main")
TASK = os.environ.get("TASK", "batch")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
USE_LLM = os.environ.get("USE_LLM", "1")
LOAD_4BIT = os.environ.get("LOAD_4BIT", "1")
INPUT_MODE = os.environ.get("INPUT_MODE", "auto")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "0")
LIMIT = os.environ.get("LIMIT", "")

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

runner_args = [
    "python", REPO_DIR / "scripts" / "kaggle_core_runner.py",
    "--task", TASK,
    "--dataset", REPO_DIR / "data" / "fraction_dataset.json",
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

run(runner_args, env=child_env)

# Only persist one file in /kaggle/working.
# Everything else stays in /tmp and will not be downloaded as kernel output.
# Write run marker so local script can avoid stale artifact downloads.
run_id = os.environ.get("RUN_ID", "")
Path(OUT_DIR, "run_id.txt").write_text(run_id)
if FINAL_ZIP.exists():
    FINAL_ZIP.unlink()
run(["bash", "-lc", f"cd {OUT_DIR} && zip -r {FINAL_ZIP} ."])
print(f"DONE: {FINAL_ZIP}", flush=True)
