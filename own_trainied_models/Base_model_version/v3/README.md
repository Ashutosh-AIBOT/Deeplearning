# v3 - Resume + Chat + Full Analysis Transformer

This is the main working version for local experimentation. It keeps the same paper architecture as v2 and adds robust checkpoint resume plus chat interfaces.

## Purpose

- Keep architecture consistent with the 2017 Transformer encoder-decoder.
- Train on DailyDialog with resumable checkpoints.
- Provide terminal chat + Gradio chat for quick testing.
- Keep rich training/data/quality/weight visualizations.

## Architecture

- Model type: Transformer encoder-decoder (seq2seq, not decoder-only LLM).
- Key defaults:
  - local mode (current laptop-safe path): all DailyDialog pairs, `N=2`, 8 epochs.
  - full mode: `N=6`, larger batch/padding for Kaggle/Colab class hardware.
- Core training recipe:
  - label smoothing
  - Adam optimizer
  - warmup/inverse-sqrt LR schedule
  - greedy decode for inference

## v3 Code-Exact Architecture Flow

Values below match the current `v3/transformer.py` code exactly:

- `make_model(...)` defaults: `N=6`, `d_model=512`, `d_ff=2048`, `h=8`, `dropout=0.1`
- active `RUN_MODE=local` config: `N=2`, `d_model=512`, `batch_size=8`, `max_padding=48`, `num_epochs=8`, `warmup=4000`
- active `RUN_MODE=full` config: `N=6`, `d_model=512`, `batch_size=32`, `max_padding=72`, `num_epochs=8`, `warmup=3000`

```mermaid
flowchart TD
    A[Source Tokens] --> B[Src Embedding<br/>d_model = 512]
    B --> C[Positional Encoding]
    C --> D[Encoder Stack xN]

    subgraph ENC[Encoder Layer (repeated N times)]
      E1[Multi-Head Self-Attention<br/>h=8, d_model=512, d_k=64]
      E2[Add + LayerNorm]
      E3[Feed-Forward<br/>512 -> 2048 -> 512]
      E4[Add + LayerNorm]
      E1 --> E2 --> E3 --> E4
    end

    D --> M[Encoder Memory]

    T[Target Tokens (shifted right)] --> U[Tgt Embedding<br/>d_model = 512]
    U --> V[Positional Encoding]
    V --> W[Decoder Stack xN]

    subgraph DEC[Decoder Layer (repeated N times)]
      D1[Masked Multi-Head Self-Attention<br/>h=8, causal mask]
      D2[Add + LayerNorm]
      D3[Cross-Attention<br/>Query=decoder, Key/Value=encoder memory]
      D4[Add + LayerNorm]
      D5[Feed-Forward<br/>512 -> 2048 -> 512]
      D6[Add + LayerNorm]
      D1 --> D2 --> D3 --> D4 --> D5 --> D6
    end

    M --> D3
    W --> G[Linear Projection to Vocab]
    G --> H[Softmax]
    H --> O[Next Token Probability]
```

`N` is set by run mode in code: local uses `N=2`, full uses `N=6`.

## Data

- Dataset: Hugging Face `roskoN/dailydialog`.
- Task: previous utterance -> next utterance.
- Typical corpus size in this project:
  - train pairs: ~76,052
  - validation pairs: ~7,069
  - test pairs: ~6,740

## Run modes

- `RUN_MODE=local` (recommended on 4 GB GPU laptop):
  - `train_pairs=None` (all train pairs)
  - `valid_pairs=None` (all validation pairs)
  - `batch_size=8`
  - `num_epochs=8`
  - `N=2`
  - `max_padding=48`
- `RUN_MODE=full` (Kaggle/Colab target):
  - `batch_size=32`
  - `N=6`
  - `max_padding=72`
  - may OOM on GTX 1650 4 GB

## Checkpoint and resume design

Current local prefix in code: `dailydialog_local_all8e_`

Artifacts:

- `...last.pt` -> latest resume point
- `...step_N.pt` -> periodic checkpoint
- `...epoch_N.pt` -> end-of-epoch checkpoint
- `...final.pt` -> finished training
- vocab file based on config (`vocab_local_all.json` / `vocab.json`)

Resume behavior:

- If `final.pt` exists: load finished model.
- Else if `last.pt` exists: continue training from saved step/epoch.
- Else: start training from scratch.

## Chat interfaces

- `chat_loop(...)` for terminal one-turn chat.
- `launch_gradio_chat(...)` for notebook UI.
- Gradio uses message format with role/content entries.
- This is still a 2017 seq2seq reply model, not ChatGPT memory chat.

## What is measured

- Training and validation loss per epoch.
- Token-level accuracy/chance comparison.
- Expected vs predicted token overlap and example comparisons.
- Parameter/weight summaries and layer-level diagnostics.

## Limitations

- Small context window compared with modern LLMs.
- Greedy decoding can produce repetitive replies.
- No instruction tuning, RLHF, or long multi-turn memory.
- Domain coverage limited by DailyDialog conversational data.

## Typical local workflow

1. Set `RUN_MODE=local`.
2. Run notebook/script.
3. Let it train; interrupt safely if needed.
4. Resume later by re-running (uses `last.pt`).
5. Evaluate metrics and test via Gradio/terminal chat.

## Role in timeline

- v3 is the mature encoder-decoder baseline with robust training operations.
- v4 is the next research stage for architecture scaling and improvements.
