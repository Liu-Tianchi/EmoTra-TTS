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
"""Step 0: Download EmoVoice-DB Dataset"""

import argparse
import os
from pathlib import Path

# Configuration
DATASET_REPO = "yhaha/EmoVoice-DB"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

parser = argparse.ArgumentParser(description="Step 0: Download EmoVoice-DB dataset")
parser.add_argument(
    "--data_dir",
    type=Path,
    default=DEFAULT_DATA_DIR,
    help="Directory to save the dataset (default: <repo>/data). "
         "The dataset is placed under <data_dir>/EmoVoice-DB.",
)
parser.add_argument(
    "-y", "--yes",
    action="store_true",
    help="Skip the interactive re-download prompt and reuse an existing dataset.",
)
args = parser.parse_args()

DATA_DIR = args.data_dir.expanduser().resolve()
EMOVOICE_DIR = DATA_DIR / "EmoVoice-DB"

print("=" * 60)
print("Step 0: Download EmoVoice-DB Dataset")
print("=" * 60)
print(f"Repository: {DATASET_REPO}")
print(f"Destination: {EMOVOICE_DIR}")
print()

# Create data directory
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Check if already exists
if EMOVOICE_DIR.exists() and any(EMOVOICE_DIR.iterdir()):
    print(f"Dataset directory already exists: {EMOVOICE_DIR}")
    if args.yes:
        response = 'n'
    else:
        response = input("Re-download? (y/n): ").strip().lower()
    if response != 'y':
        print("Using existing dataset.")
        # Show what's there
        print("\nExisting files:")
        for item in sorted(EMOVOICE_DIR.iterdir())[:10]:
            print(f"  - {item.name}")
        print("\n" + "=" * 60)
        print("Step 0 Complete!")
        print("=" * 60)
        exit(0)

print("Using Python API to download repository")
print("This will download all files including audio (~several GB)")
print()

try:
    from huggingface_hub import snapshot_download
    
    # Set mirror for China
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    print("Downloading...")
    print("(This may take a while, please be patient)")
    print()
    
    # Download entire repository
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=str(EMOVOICE_DIR),
        local_dir_use_symlinks=False,
        resume_download=True
    )
    
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)
    
except ImportError:
    print("\nError: huggingface_hub not installed")
    print("\nPlease install it with:")
    print("  pip install huggingface_hub")
    exit(1)
    
except Exception as e:
    print(f"\nError during download: {e}")
    print("\n" + "=" * 60)
    print("Alternative: Manual Download")
    print("=" * 60)
    print("1. Visit: https://huggingface.co/datasets/yhaha/EmoVoice-DB")
    print("2. Click 'Files and versions'")
    print("3. Download files manually or use git:")
    print(f"   git clone https://huggingface.co/datasets/{DATASET_REPO} {EMOVOICE_DIR}")
    print(f"   cd {EMOVOICE_DIR}")
    print("   git lfs pull")
    exit(1)

# Check downloaded content
print("\nChecking downloaded files...")
print(f"Location: {EMOVOICE_DIR}")

# Count files
audio_files = list(EMOVOICE_DIR.rglob("*.wav"))
json_files = list(EMOVOICE_DIR.glob("*.jsonl"))

print(f"\nFiles found:")
print(f"  - JSONL files: {len(json_files)}")
for f in json_files:
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f"    - {f.name} ({size_mb:.2f} MB)")

print(f"  - Audio files (.wav): {len(audio_files)}")

zip_files = list((EMOVOICE_DIR / "audio").glob("*.zip")) if (EMOVOICE_DIR / "audio").exists() else []

if len(audio_files) > 0:
    print(f"\nSample audio files:")
    for f in sorted(audio_files)[:5]:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    - {f.relative_to(EMOVOICE_DIR)} ({size_mb:.2f} MB)")
    
    total_size = sum(f.stat().st_size for f in audio_files)
    print(f"\nTotal audio size: {total_size / (1024 ** 3):.2f} GB")
elif len(zip_files) > 0:
    print(f"\nAudio archives (.zip): {len(zip_files)} (one per emotion)")
    for f in sorted(zip_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    - audio/{f.name} ({size_mb:.2f} MB)")
    print("\nNote: audio ships as per-emotion zip archives. Extract them with:")
    print("  python step0_prepare_dataset.py")
else:
    print("\n⚠ Warning: No audio files or archives found!")
    print("Audio files may still be downloading or need git-lfs")

print("\n" + "=" * 60)
print("Step 0 Complete!")
print("=" * 60)
print(f"\nDataset location: {EMOVOICE_DIR}")
print("\nNext step: extract the audio archives with step0_prepare_dataset.py")
