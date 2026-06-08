# ---------------------------------------------------------------------
# Export artifacts to Kaggle downloadable output.
# IMPORTANT:
# Kaggle CLI can only download files that are exposed as kernel outputs.
# So we explicitly copy everything to /kaggle/working/outputs and create
# /kaggle/working/exact_artifacts.zip.
# ---------------------------------------------------------------------
import os
import shutil
import subprocess
from pathlib import Path

KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_OUTPUTS = KAGGLE_WORKING / "outputs"
KAGGLE_ZIP = KAGGLE_WORKING / "exact_artifacts.zip"

KAGGLE_OUTPUTS.mkdir(parents=True, exist_ok=True)

# OUT_DIR should already exist in runner.py.
# It may be a string or Path depending on your current code.
src_out = Path(OUT_DIR)

print(f"[artifact] src_out = {src_out}")
print(f"[artifact] src_out exists = {src_out.exists()}")

if src_out.exists():
    for item in src_out.iterdir():
        dest = KAGGLE_OUTPUTS / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

# Always write run marker.
run_id = os.environ.get("RUN_ID", "")
(KAGGLE_OUTPUTS / "run_id.txt").write_text(run_id, encoding="utf-8")

# Debug listing before zip.
print("[artifact] files prepared in /kaggle/working/outputs:")
subprocess.run(
    "find /kaggle/working/outputs -maxdepth 4 -type f -print",
    shell=True,
    check=False,
)

# Create zip at exact Kaggle working root.
if KAGGLE_ZIP.exists():
    KAGGLE_ZIP.unlink()

subprocess.run(
    "cd /kaggle/working/outputs && zip -r /kaggle/working/exact_artifacts.zip .",
    shell=True,
    check=True,
)

print("[artifact] final files in /kaggle/working:")
subprocess.run(
    "find /kaggle/working -maxdepth 3 -type f -print",
    shell=True,
    check=False,
)

print(f"DONE: {KAGGLE_ZIP}")