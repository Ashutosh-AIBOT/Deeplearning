#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="transformer-env"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMPDIR_PATH="${HOME}/bigtmp"

echo "Using project directory: ${PROJECT_DIR}"
mkdir -p "${TMPDIR_PATH}" "${TMPDIR_PATH}/cache"

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | rg -q "^${ENV_NAME}[[:space:]]"; then
  echo "Conda env '${ENV_NAME}' already exists."
else
  echo "Creating conda env '${ENV_NAME}'..."
  conda create -y -n "${ENV_NAME}" python=3.12
fi

echo "Installing PyTorch CUDA 12.8 build..."
TMPDIR="${TMPDIR_PATH}" conda run -n "${ENV_NAME}" pip install \
  --cache-dir "${TMPDIR_PATH}/cache" \
  torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

echo "Installing remaining dependencies..."
TMPDIR="${TMPDIR_PATH}" conda run -n "${ENV_NAME}" pip install \
  --cache-dir "${TMPDIR_PATH}/cache" \
  -r "${PROJECT_DIR}/requirements.txt"

echo "Installing spaCy English model..."
TMPDIR="${TMPDIR_PATH}" conda run -n "${ENV_NAME}" python -m spacy download en_core_web_sm

echo "Verifying versions..."
conda run -n "${ENV_NAME}" python -c "import sys, torch, pandas, altair, datasets, spacy, tokenizers, GPUtil; print('Python:', sys.version); print('torch:', torch.__version__); print('pandas:', pandas.__version__); print('altair:', altair.__version__); print('datasets:', datasets.__version__); print('spacy:', spacy.__version__); print('tokenizers:', tokenizers.__version__); print('GPUtil:', GPUtil.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU visible to torch')"

echo "Done. Activate with: conda activate ${ENV_NAME}"
