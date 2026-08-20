# GPT-v1 — Decoder-only Transformer Language Model (research / learning project)

A from-scratch GPT-1-style decoder-only language model: pretraining (next-token prediction) on DailyDialog + Alpaca + GSM8K, with a complete research pipeline — configs per run, checkpointing with rollback, resume, TensorBoard, EDA, quality dashboards, architecture graphs (torchviz / netron), and a Gradio chat UI.

> **⚠️ Read `../missing.md` too** — it is the detailed post-mortem + the phase-by-phase blueprint for GPT-2. This README is the project card + the issue ledger; `missing.md` is the full manual.

---

## 1. Quick Facts

| Item | Value |
|---|---|
| Architecture | GPT-1 style decoder-only (causal masked attention, tied LM head w/ final LN) |
| Size | **40,645,632 params (~40.6 M)** |
| Config (local) | n_layer=8, n_embd=512, n_head=8, d_ff=2048, ctx=128, stride=96 |
| Tokenizer | WordLevel, 30,000 tokens (special: `<pad>` `<bos>` `<eos>` `<unk>`) |
| Data | DailyDialog (76,026) + Alpaca (52,000) + GSM8K (7,000) = **135,026 train texts**; valid 8,069 |
| Train samples | 7,882 LM batches/epoch × 8 = 63,056 samples; 5 epochs ≈ 315,280 samples ≈ **40.4 M tokens** |
| Final checkpoint | `gptv1_local_final.pt` — epoch 4, step 4925, **val_loss 6.05** (ppl ≈ 425, heavily under-trained) |
| Hardware / speed | GTX 1650 4 GB, ~5,400 tok/s fs32 → ~25 min/epoch |
| Runtime | ~2 h for 5 epochs |
| Best quality metric | Teacher-forced token acc 37.89 % (V4 run, similar regime); chance ≈ 1/30000 = 0.003 % |

---

## 2. What This Project Does (pipeline map)

```
build texts → vocab (WordLevel) → LM blocks (ctx 128, stride 96)
→ pretrain 5 epochs (AdamW, cosine LR, warmup 2000, grad accum 8, clip 1.0)
→ checkpoint every 2000 steps + per epoch + final (keep 3 newest, rollback helper)
→ valid ppl + token accuracy → charts (altair) → weight dashboard
→ torchviz graph / netron / TensorBoard → zero-shot samples → Gradio chat
```

Run everything from the notebook: **`transformer_FINAL_FULL_v4.ipynb`** (31 cells; restarts → loads `final.pt`, never retrains). Original wrapper `transformer.ipynb` is preserved and never deleted.

---

## 3. How to Use

### Chat
```python
# notebook (Part 10 auto-launches now since v4)
import os; os.environ["SKIP_CHAT"] = "1"   # only if you DON'T want the UI
# then re-run last cell → http://127.0.0.1:7860
```

### From a terminal
```bash
# launch chat server with the trained model (keeps running; Ctrl-C to stop)
python /tmp/opencode/chat_server.py   # or use launch_gradio_chat directly

# one-shot reply
python -c "import transformer as g; v=g.build_vocab([],'gptv1_vocab_local.json');
import torch; c=torch.load('gptv1_local_final.pt',map_location='cpu');
m=g.make_model(v.get_vocab_size(),c['config']); m.load_state_dict(c['model']);
m=m.cuda().eval(); print(g.generate_reply(m,v,'hello'))"
```

### Files
| File | Purpose |
|---|---|
| `transformer.py` | Source of truth: architecture + pipeline + chat (fixed: GSM8K HF loader, Gradio 6, CUDA RNG restore, prompt-echo-free replies, repetition penalty) |
| `transformer_FINAL_FULL_v4.ipynb` | Full research notebook (use this) |
| `transformer.ipynb` | Original wrapper — permanent, never delete |
| `gptv1_vocab_local.json` | WordLevel tokenizer (trained on the 3-dataset mix) |
| `gptv1_local_final.pt` / `last.pt` / `epoch_N.pt` / `step_N.pt` | Checkpoints (final, epoch 2–4, step 2000/4000) |
| `runs/gptv1_local/` | TensorBoard logs |
| `../missing.md` | Post-mortem + blueprint (READ BEFORE GPT-2) |

---

## 4. Issue Ledger — what went wrong, and the fix target for GPT-2

> These are the issues from THIS run, in priority order. GPT-2 must not repeat them. Full detail in `../missing.md`.

| # | Issue (evidence from this run) | Root cause | Fix in GPT-2 |
|---|---|---|---|
| 1 | Chat replies are incoherent / don't answer | **Zero-shot LM — no chat/instruction fine-tuning** | SFT on masked (prompt → response) pairs, loss only on `Assistant:` tokens, LR 1e-5, 1–3 epochs (LoRA optional) |
| 2 | 21 % of generated tokens are `<unk>` (measured: 6/28), model babbles them | WordLevel tokenizer (30 k), `min_frequency=2`; unknown words get one embedding and the LM learns to emit it as filler | **Byte-level BPE** (GPT-2's own tokenizer), 32 k vocab → `<unk>` impossible; verify unk-rate=0 on train+valid |
| 3 | "Again and again" word loops (`the / a / to / and`) | Decoding without penalties; model's distribution dominated by high-frequency tokens; repetitive training data worsens it | Sampling order: penalties → temperature → top-p, stop sequences (`\n`, `User:`, `<eos>`); frequency_penalty 0.3–0.7 for loops |
| 4 | Replies echo `User: … Assistant:` (fixed in v4 at `transformer.py` `generate_reply`) | decode included the prompt tokens | Larger project norm: always test prompt-echo-free decoding (regression test on the eval set) |
| 5 | val_loss 6.05 — under-trained, not overfitted (train ≈ valid loss) | ~40.4 M tokens for 40.6 M params = **1 token/param** (Chinchilla: 20) | Train 12–24 h (≈155–466 M tokens); or shrink model; use AMP (2× speed) + ctx ≥ 256 |
| 6 | ctx = 128 — replies have no conversation memory | Dialog needs longer context for coherence | 512 via mixed precision; 1024 if budget allows |
| 7 | Curly-quote / spacing noise (`I ’ t`) leaked into training text | Raw data unnormalized | NFKC unicode normalize, straight quotes, whitespace collapse before tokenizer |
| 8 | Template seams in output ("Assistant: Assistant:") | Three task types (dialog / instruction / math) merged into one `User:/Assistant:` template | Clean + split data by task; per-task format; versioned datasets |
| 9 | No generated-text evaluation — trained and shipped blind | Only ppl + teacher-forced accuracy existed | Frozen 50–100 prompt eval set + distinct-1/2 + `<unk>` rate + human 1–5 score, logged per experiment |
| 10 | fp32 on 4 GB limits everything | No mixed precision | `torch.autocast` (half precision) + keep grad clip 1.0 |

### What this project did RIGHT (keep for GPT-2)
- ✅ Checkpointing with rollback + resume (`last.pt`/`final.pt`/step files, keep-3 cleanup)
- ✅ CUDA-safe RNG state save/restore (fix: `.cpu().to(torch.uint8)`, slice by device count)
- ✅ Gradio 6 compatibility, inline chat, prompt-state saved
- ✅ Reproducible per-run config + TensorBoard + altair charts
- ✅ No retraining when `final.pt` exists (fast reloads)

---

## 5. Reusable Code Notes (map to GPT-2)

| Topic | Where | GPT-2 action |
|---|---|---|
| Architecture class | `transformer.py` `GPTDecoderOnly` | Add GPT-2 deltas: pre-norm LN inside blocks, residual scaling `1/√N`, final LN + activation scale, bigger ctx |
| Byte-level BPE | (none — WordLevel today) | Implement/import `tiktoken` or `minbpe`; it is byte-level by design |
| Data cleaning/dedup | (none) | Add Phase-2 stage (see `../missing.md` §5) |
| Sampling knobs | `generate()` + `generate_reply()` | Extend with top-p, min-p, frequency/presence penalties in the spec order |
| Eval | `evaluate_loss()` + accuracy helper | Add frozen-prompt eval + distinct-n + human scoring (one-var-at-a-time log) |

---

## 6. Baseline Numbers (so GPT-2 has something to beat)

| Metric | GPT-v1 (this model) |
|---|---|
| Tokenizer `<unk>` rate (generation) | ~21 % |
| Val loss (final, epoch 4) | 6.047 |
| Val ppl (capped at 20) | ≈ 20 (capped) / true ≈ 425 |
| Teacher-forced token accuracy | 37.89 % (V4-like regime) |
| Human 1–5 chat rating | Not measured — run it in GPT-2 |
| Repetition (distinct-1) | Not measured — run it in GPT-2 |

**GPT-2 success bar (suggested):** unk-rate 0, val_loss < 5.0, distinct-1 > 0.35, human smalltalk score ≥ 3.5/5. Track every experiment in a table.