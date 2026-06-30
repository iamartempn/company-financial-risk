#!/usr/bin/env python3
"""Apply the next scheduled patch."""
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

SCHEDULED_DIR = Path(".scheduled")
SCHEDULE_FILE = SCHEDULED_DIR / "schedule.json"
PROGRESS_FILE = SCHEDULED_DIR / "progress.json"
PATCHES_DIR = SCHEDULED_DIR / "patches"


def load_json(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    schedule = load_json(SCHEDULE_FILE)
    if not schedule:
        print("No schedule found.")
        sys.exit(0)

    progress = load_json(PROGRESS_FILE) or {"applied": []}
    applied = set(progress["applied"])
    today = date.today().isoformat()

    github_output = os.environ.get("GITHUB_OUTPUT", "")

    for entry in schedule:
        patch_id = entry["patch_dir"]
        if patch_id in applied:
            continue
        if entry["date"] > today:
            break

        print(f"Applying patch {patch_id}: {entry['message']}")
        patch_path = PATCHES_DIR / patch_id

        for file_info in entry["files"]:
            src = patch_path / file_info["src"]
            dst = Path(file_info["dst"])
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            subprocess.run(["git", "add", str(dst)], check=True)
            print(f"  {file_info['src']} -> {file_info['dst']}")

        progress["applied"].append(patch_id)
        save_json(PROGRESS_FILE, progress)
        subprocess.run(["git", "add", str(PROGRESS_FILE)], check=True)
        subprocess.run(["git", "commit", "-m", entry["message"]], check=True)

        if github_output:
            with open(github_output, "a") as f:
                f.write("pushed=true\n")
        return

    print("No patches due today.")
    if github_output:
        with open(github_output, "a") as f:
            f.write("pushed=false\n")


if __name__ == "__main__":
    main()
