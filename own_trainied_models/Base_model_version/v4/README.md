# v4 - Next Architecture Research Stage

This folder is the new active stage after v3. The intent is to push model quality within local hardware limits first, then scale on Kaggle/Colab when needed.

## Current status

- `v4/transformer.py` currently mirrors the v3-style codebase and run structure.
- It already includes run modes, DailyDialog loading, checkpointing, and chat utilities.
- Treat this as the experimentation branch for architecture and context upgrades.

## Main goal for v4

- Improve chatbot quality beyond v3 while staying reproducible.
- Increase useful capacity and context carefully so the laptop still runs.
- Keep a clear audit trail of every experiment and metric.

## Suggested research upgrades (laptop first)

1. Increase effective context safely:
   - test `max_padding` increments (for example 48 -> 64).
   - monitor OOM and tokens/sec impact.
2. Increase depth/width in controlled steps:
   - `N=2 -> 3/4` (first)
   - then test `d_model` growth only if memory allows.
3. Improve tokenizer/data pipeline:
   - keep DailyDialog base, optionally add clean conversational corpora.
4. Improve decoding/evaluation:
   - compare greedy vs constrained sampling in a controlled notebook branch.

## Recommended experiment matrix

- Keep one variable change per run.
- Example sequence:
  - baseline: `N=2, max_padding=48`
  - run A: `N=3, max_padding=48`
  - run B: `N=2, max_padding=64`
  - run C: `N=3, max_padding=64` (if stable)

Track for each run:

- train/valid loss
- token accuracy
- training speed (tokens/sec)
- GPU memory use and stability
- qualitative chat samples

## Hardware and run guidance

- Local laptop (GTX 1650 4 GB):
  - avoid sudden jumps to `N=6` + large batch sizes.
  - keep batch size conservative and use gradient accumulation.
- Kaggle/Colab:
  - use for larger architecture/context once local tuning is validated.

## Limitations to remember

- This architecture is still encoder-decoder seq2seq, not a modern multi-billion decoder-only LLM.
- Better quality is possible, but there is a hard cap from:
  - model size
  - context window
  - dataset scope
  - available compute

## Development rule for v4

- Do not delete older versions.
- Keep v1/v2/v3 as frozen references.
- Document every major v4 configuration and outcome in this folder as experiments progress.
