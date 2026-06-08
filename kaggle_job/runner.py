import os
import shutil
import subprocess
from pathlib import Path

REPO_URL = os.environ.get("REPO_URL", "https://github.com/Huoijo/AI_Logic_EXACT.git")
TASK = os.environ.get("TASK", "batch")
INPUT_MODE = os.environ.get("INPUT_MODE", "auto")
BATCH_SIZE = os.environ.get("BATCH_SIZE", "0")
LIMIT = os.environ.get("LIMIT", "")
RUN_ID = os.environ.get("RUN_ID", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
ADAPTER_PATH = os.environ.get("ADAPTER_PATH", "")
DATASET = os.environ.get("DATASET", "data/fraction_dataset.json")
LOG_EACH_CASE = os.environ.get("LOG_EACH_CASE", "1")

WORK_DIR = Path("/kaggle/working")
REPO_DIR = WORK_DIR / "AI_Logic_EXACT"
OUT_DIR = WORK_DIR / "outputs"
ZIP_PATH = WORK_DIR / "exact_artifacts.zip"


def run(cmd, cwd=None, env=None):
    print("RUN:", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def run_shell(cmd, cwd=None):
    print("RUN:", cmd, flush=True)
    subprocess.run(cmd, cwd=cwd, shell=True, check=True)


def main():
    print("=" * 70, flush=True)
    print("EXACT Kaggle Runtime", flush=True)
    print(f"TASK={TASK}", flush=True)
    print(f"MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"ADAPTER_PATH={ADAPTER_PATH}", flush=True)
    print(f"INPUT_MODE={INPUT_MODE}", flush=True)
    print(f"DATASET={DATASET}", flush=True)
    print(f"BATCH_SIZE={BATCH_SIZE}", flush=True)
    print(f"LIMIT={LIMIT}", flush=True)
    print(f"LOG_EACH_CASE={LOG_EACH_CASE}", flush=True)
    print(f"RUN_ID={RUN_ID}", flush=True)
    print("=" * 70, flush=True)

    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])

    req_kaggle = REPO_DIR / "requirements_kaggle.txt"
    req_local = REPO_DIR / "requirements.txt"
    if req_kaggle.exists():
        run(["python", "-m", "pip", "install", "-q", "-r", str(req_kaggle)])
    elif req_local.exists():
        run(["python", "-m", "pip", "install", "-q", "-r", str(req_local)])

    dataset_path = REPO_DIR / DATASET
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}. Commit and push {DATASET} first.")

    env = os.environ.copy()
    env["MODEL_NAME"] = MODEL_NAME
    env["RUN_ID"] = RUN_ID
    env["INPUT_MODE"] = INPUT_MODE
    env["LOG_EACH_CASE"] = LOG_EACH_CASE
    if ADAPTER_PATH:
        # Adapter may be committed in repo or copied into Kaggle model/dataset later.
        maybe_repo_adapter = REPO_DIR / ADAPTER_PATH
        env["ADAPTER_PATH"] = str(maybe_repo_adapter if maybe_repo_adapter.exists() else ADAPTER_PATH)

    cmd = [
        "python", str(REPO_DIR / "scripts" / "kaggle_core_runner.py"),
        "--task", TASK,
        "--dataset", str(dataset_path),
        "--input-mode", INPUT_MODE,
        "--out", str(OUT_DIR),
    ]
    if BATCH_SIZE and str(BATCH_SIZE) != "0":
        cmd += ["--batch-size", str(BATCH_SIZE)]
    if LIMIT:
        cmd += ["--limit", str(LIMIT)]
    if str(LOG_EACH_CASE) == "1":
        cmd += ["--log-cases"]

    run(cmd, cwd=REPO_DIR, env=env)

    (OUT_DIR / "run_id.txt").write_text(RUN_ID, encoding="utf-8")
    print("[artifact] files prepared in /kaggle/working/outputs:", flush=True)
    run_shell("find /kaggle/working/outputs -maxdepth 5 -type f -print")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    run_shell("cd /kaggle/working/outputs && zip -r /kaggle/working/exact_artifacts.zip .")
    print("[artifact] final files in /kaggle/working:", flush=True)
    run_shell("find /kaggle/working -maxdepth 3 -type f -print")
    print(f"DONE: {ZIP_PATH}", flush=True)


if __name__ == "__main__":
    main()
