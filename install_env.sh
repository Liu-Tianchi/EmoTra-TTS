#!/usr/bin/env bash
#
# One-command installer for the EmoTra-TTS main conda environment ("EmoTra").
#
# This wraps `conda env create` with a pip build constraint (constraints.txt)
# that is required to build openai-whisper==20231117 (see constraints.txt for
# the full explanation). Using this script means a fresh clone installs in a
# single step.
#
# Usage:
#   bash install_env.sh
#   conda activate EmoTra
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PIP_CONSTRAINT="${SCRIPT_DIR}/constraints.txt"

conda env create -f "${SCRIPT_DIR}/environment_emotra.yml"

echo
echo "EmoTra environment created. Activate it with:"
echo "    conda activate EmoTra"
