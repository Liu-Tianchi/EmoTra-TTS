#!/usr/bin/env python3
# Copyright (c) 2026 Tianchi Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Step 0 (prepare): Extract the EmoVoice-DB audio archives.

`step0_download_emovoice.py` downloads the dataset, but the audio ships as one
zip archive per emotion under ``<data_dir>/EmoVoice-DB/audio/`` (angry.zip,
happy.zip, ...). The rest of the pipeline (e.g. ``step1_filter_by_vad.py``)
expects the extracted layout ``<data_dir>/EmoVoice-DB/audio/<emotion>/*.wav``,
matching the ``target_wav`` paths in the ``*.jsonl`` files.

This script extracts each archive in place (idempotent) and reports the result.
"""

import argparse
import zipfile
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

parser = argparse.ArgumentParser(
    description="Step 0: Extract EmoVoice-DB audio archives into per-emotion folders."
)
parser.add_argument(
    "--data_dir",
    type=Path,
    default=DEFAULT_DATA_DIR,
    help="Directory that holds the datasets (default: <repo>/data). "
         "The dataset is expected under <data_dir>/EmoVoice-DB.",
)
parser.add_argument(
    "--force",
    action="store_true",
    help="Re-extract even if the target emotion folder already exists.",
)
args = parser.parse_args()

DATA_DIR = args.data_dir.expanduser().resolve()
EMOVOICE_DIR = DATA_DIR / "EmoVoice-DB"
AUDIO_DIR = EMOVOICE_DIR / "audio"

print("=" * 60)
print("Step 0: Prepare EmoVoice-DB (extract audio archives)")
print("=" * 60)
print(f"Dataset: {EMOVOICE_DIR}")
print()

if not AUDIO_DIR.is_dir():
    print(f"Error: audio directory not found: {AUDIO_DIR}")
    print("Run step0_download_emovoice.py first (optionally with --data_dir).")
    raise SystemExit(1)

zip_files = sorted(AUDIO_DIR.glob("*.zip"))
if not zip_files:
    print(f"No .zip archives found in {AUDIO_DIR}.")
    print("Nothing to extract — the dataset may already be prepared.")

total_wavs = 0
for zip_path in zip_files:
    emotion = zip_path.stem  # e.g. "angry"
    target_dir = AUDIO_DIR / emotion

    if target_dir.is_dir() and any(target_dir.glob("*.wav")) and not args.force:
        n = len(list(target_dir.glob("*.wav")))
        print(f"  [skip] {emotion}: already extracted ({n} wav files)")
        total_wavs += n
        continue

    print(f"  [extract] {zip_path.name} -> {target_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            # Skip macOS resource-fork junk that ships inside these archives.
            if member.startswith("__MACOSX/") or "/._" in member or member.endswith("/._"):
                continue
            zf.extract(member, AUDIO_DIR)

    n = len(list(target_dir.glob("*.wav"))) if target_dir.is_dir() else 0
    print(f"           {n} wav files")
    total_wavs += n

print()
print("=" * 60)
print("Step 0 (prepare) Complete!")
print("=" * 60)
print(f"Audio root: {AUDIO_DIR}")
print(f"Emotion folders: {', '.join(p.name for p in sorted(AUDIO_DIR.iterdir()) if p.is_dir())}")
print(f"Total wav files: {total_wavs}")
