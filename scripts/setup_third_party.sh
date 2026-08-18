#!/usr/bin/env bash
#
# Fetch the third-party dependencies that EmoTra-TTS does NOT redistribute.
#
# This repository ships an (almost) empty `third_party/` directory on purpose:
# every external component is fetched here, from its original upstream, under its
# own license. Run this once after cloning.
#
#   bash scripts/setup_third_party.sh
#
# What it fetches:
#   1. Matcha-TTS            (git submodule)          — Flow-matching runtime dep
#   2. UniSpeech SV source   (github.com/microsoft/UniSpeech) — SIM evaluation
#   3. s3prl WavLM source    (github.com/s3prl/s3prl) — SIM evaluation (Torch Hub cache)
#   4. WavLM-large SV ckpt   (~1.3 GB, Google Drive)  — SIM evaluation
#
# The w2v2-vad ONNX model used by Step 1 is downloaded automatically on first use
# (from Zenodo) into third_party/models/, so it is not handled here.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "==> [1/4] Matcha-TTS submodule"
git submodule update --init --recursive

echo "==> [2/4] UniSpeech speaker-verification source (SIM eval)"
UNI_DST="third_party/seed-tts-eval/thirdparty/UniSpeech"
if [ -d "${UNI_DST}/downstreams/speaker_verification" ]; then
    echo "    already present, skipping"
else
    tmp="$(mktemp -d)"
    git clone --depth 1 https://github.com/microsoft/UniSpeech "${tmp}/UniSpeech"
    mkdir -p "${UNI_DST}"
    cp -r "${tmp}/UniSpeech/downstreams" "${UNI_DST}/"
    rm -rf "${tmp}"
    echo "    -> ${UNI_DST}/downstreams/speaker_verification"
fi

echo "==> [3/4] s3prl WavLM source (Torch Hub cache)"
S3PRL_DST="${HOME}/.cache/torch/hub/s3prl_s3prl_main"
if [ -d "${S3PRL_DST}" ]; then
    echo "    already present, skipping"
else
    git clone https://github.com/s3prl/s3prl "${S3PRL_DST}"
fi

echo "==> [4/4] WavLM-large SV checkpoint (~1.3 GB, Google Drive)"
CKPT="third_party/seed-tts-eval/pretrained_models/wavlm_large_finetune.pth"
if [ -f "${CKPT}" ]; then
    echo "    already present, skipping"
else
    mkdir -p "$(dirname "${CKPT}")"
    if command -v gdown >/dev/null 2>&1; then
        gdown 1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP -O "${CKPT}"
    else
        echo "    !! gdown not found. Install it (pip install gdown) or run this in the 'sim' env,"
        echo "       then re-run this script, or download manually:"
        echo "       https://drive.google.com/file/d/1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP/view"
        echo "       -> ${CKPT}"
    fi
fi

echo
echo "third_party/ setup complete."
