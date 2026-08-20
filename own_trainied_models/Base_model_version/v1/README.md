# v1 - Base Annotated Transformer Prototype

This folder is the earliest prototype. It starts from a Colab-style copy of the Harvard Annotated Transformer code and then adapts it toward DailyDialog training.

## Purpose

- Establish the original paper architecture in your own project.
- Move from `Multi30k` comments/examples to DailyDialog next-utterance training.
- Produce a first train/save baseline before resume and chat UI improvements.

## Architecture

- Model type: Transformer encoder-decoder (Attention Is All You Need, 2017 style).
- Core blocks:
  - token embeddings + sinusoidal positional encoding
  - multi-head self-attention
  - encoder-decoder (cross) attention
  - position-wise feed-forward network
  - residual connections + layer normalization
- Typical paper-size hyperparameters in this file:
  - `N=6` layers
  - `d_model=512`
  - `d_ff=2048`
  - `h=8` heads
- Optimizer/schedule: Adam + warmup schedule (`rate()` pattern).

## Data

- Dataset: Hugging Face `roskoN/dailydialog`.
- Task formulation: previous utterance -> next utterance (seq2seq pair).
- Earlier `Multi30k` code remains mostly as comments/history in this version.

## Outputs in v1

- Vocabulary file: `vocab.json`
- Model checkpoint naming in this stage:
  - `dailydialog_model_*.pt`
  - `dailydialog_model_final.pt`

## Limitations

- Colab/export style script with notebook shell commands embedded.
- Less clean run-mode separation compared to later versions.
- No robust resume workflow.
- No Gradio chat interface.
- Mixed historical sections from original tutorial + adapted sections.

## Typical run flow

1. Install dependencies and spaCy model.
2. Build DailyDialog pairs.
3. Build vocabulary.
4. Create dataloaders.
5. Train for configured epochs.
6. Save final checkpoint.

## Role in project timeline

- v1 is your base learning milestone: "I implemented and trained the classic Transformer pipeline."
- Use v2/v3 for cleaner experimentation and reproducible local runs.
