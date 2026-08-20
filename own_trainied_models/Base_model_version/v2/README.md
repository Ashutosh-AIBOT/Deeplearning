# v2 - DailyDialog Training and Save Pipeline

This version is the cleaned training pipeline after v1. It keeps the paper Transformer architecture and focuses on train/save quality analysis (no live chat UI).

## Purpose

- Keep architecture faithful to the 2017 encoder-decoder Transformer.
- Train on DailyDialog as a next-utterance task.
- Support two execution modes (`local` and `full`).
- Add stronger visualization/analysis for data, loss, and quality.

## Architecture

- Model type: Transformer encoder-decoder (paper style).
- Default architecture choices:
  - local mode: smaller (`N=2`) for laptop iteration
  - full mode: paper-style depth (`N=6`) for bigger hardware
- Shared core components:
  - embeddings + sinusoidal positional encoding
  - masked decoder self-attention
  - encoder-decoder attention
  - feed-forward layers
  - label smoothing + Adam + warmup LR schedule

## Data

- Dataset: `roskoN/dailydialog` from Hugging Face (dataset only).
- Training pairs: previous line -> next line.
- Modes:
  - `RUN_MODE=local`: subset for quick local training.
  - `RUN_MODE=full`: full dataset for long run / Kaggle/Colab.

## Environment and dependencies

- Main files:
  - `transformer.py`
  - `transformer.ipynb`
  - `requirements.txt`
  - `setup_conda.sh`
- Local env target: `transformer-env` (Python 3.12 + CUDA torch).

## Outputs and artifacts

- Model checkpoints:
  - `chatmodel-v2.pt`
  - `chatmodel-v2-full.pt` (full run)
- Vocabulary:
  - `vocab.json`
- Visualization includes:
  - dataset views and token/length distribution
  - training curves (loss/LR/progress)
  - quality comparisons (expected vs predicted, overlap)
  - weight/parameter dashboards

## Limitations

- No interactive chat endpoint in v2.
- Decoder outputs can still be generic on smaller local runs.
- Quality depends heavily on run mode, epochs, and GPU time.

## Step-by-step run flow

1. Set `RUN_MODE` (`local` for laptop, `full` for stronger compute).
2. Load DailyDialog and build pairs.
3. Build/load vocabulary.
4. Train model for configured epochs.
5. Save checkpoint + vocab.
6. Review quality and training charts.

## Role in timeline

- v2 is the clean train/save baseline.
- v3 extends this baseline with resume checkpoints and chat UI.
