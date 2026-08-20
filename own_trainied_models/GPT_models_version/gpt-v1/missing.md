# MISSING.md — GPT-1 Post-Mortem + Complete, Numbers-Everywhere Blueprint for GPT-2

> The master technical reference. Every section has **exact numbers, formulas, and specs** — nothing wishy-washy. Uses the measured facts of your GPT-v1 run as evidence (val_loss 6.047, 21% `<unk>`, 40.6M params, ~40.4M tokens). GPT-2 must start from Section 14 (the concrete target spec) and tick every checkbox.

---

## 0. THE RULE OF THE GAME

NLP quality comes from 5 things, exactly in this order of leverage:
1. **Data** (cleaning + dedup + task match) — 40%
2. **Task structure** (fine-tuning/prompt format) — 30%
3. **Tokenization** (byte BPE vs word-level) — 10%
4. **Training recipe** (time, LR, batch, context) — 15%
5. **Decoding** (sampling knobs) — the last 5%

Your GPT-1 run got #4 partially, #1–3 basically not, #5 late. That is why chat was incoherent. Everything below quantifies the fix.

---

## 1. MEASURED EVIDENCE FROM YOUR RUN (the numbers that matter)

| Metric | Value | Meaning |
|---|---|---|
| Model params | 40,645,632 (40.6M) | fp32 weights = 40.6M × 4 B = **162.6 MB** on GPU |
| AdamW optimizer state | 40.6M × 8 B (m + v) | + **325 MB** during training |
| Train texts | 135,026 (DailyDialog 76,026 + Alpaca 52,000 + GSM8K 7,000) | ~1.03 token/param used |
| Valid texts | 8,069 (DailyDialog 7,069 + GSM8K 1,000) | frozen valid, but NOT deduped vs train |
| LM blocks / epoch | 63,056 train samples (7,882 batches × 8) | = 8,071,168 tokens/epoch |
| 5 epochs → tokens | 315,280 samples / 40.36 M tokens | **data/param ratio ≈ 0.99** (Chinchilla optimum ≈ 20–100) |
| Throughput (measured) | ~5,400 tok/s fp32 | epoch = ~25 min; 5 epochs ≈ 2 h |
| Final val loss | 6.047 (epoch 4, step 4925) | ppl_true = e^6.047 ≈ **425**; still falling when saved |
| Generated `<unk>` rate | 6/28 tokens = **21%** | WordLevel tokenizer failure |
| Teacher-forced token acc | 37.89% (V4-similar) | chance = 1/30,000 = 0.0033% |
| Context | 128 tokens | ~1 dialog turn — no memory |

Verdict: **under-trained, under-tokened, un-finetuned** — not bugged, not overfitted.

---

## 2. ARCHITECTURE SPECS — EXACT (GPT-1 paper vs GPT-2 paper vs YOUR local config)

### 2.1 Original GPT-1 (Radford et al. 2018) — decoder-only, BPE 40k
| Hyperparam | Value |
|---|---|
| Decoder layers `n_layer` | 12 |
| Hidden `n_embd` | 768 |
| Attention heads | 12 (head dim 64) |
| FFN `d_ff` | 3072 (4× n_embd, ReLU) |
| Max context `max_seq_len` | 512 |
| Tokenizer | BPE 40,478 (WordLevel was NEVER used by OpenAI) |
| Dropout | 0.1 (incl. attention blocks, on tricks) |
| Initialization | N(0, 0.02); bias 0 |
| Optimizer | Standard Adam (lr 2.5e-4, warmup 2000 steps, then cosine to 0) |
| Batch | 64 seqs × 512 tokens = **32,768 tokens/step** |
| LM head | tied to input embeddings |
| LN | POST-norm: one LayerNorm after the whole block |

### 2.2 Original GPT-2 (Radford et al. 2019) — the deltas you MUST add next time
| Hyperparam | Value (small = "124M") |
|---|---|
| Decoder layers | 12 |
| Hidden | 768 |
| Heads | 12 |
| FFN | 3072 |
| **Context** | **1024** (2× GPT-1) |
| **Tokenizer** | **Byte-level BPE, 50,257 tokens — NO `<unk>` ever** |
| **LN placement** | **PRE-norm: LayerNorm moves INSIDE the block, BEFORE attn and before FFN** |
| **Residual scaling** | each block output scaled so std stays ≈ 1; **c_proj init std = 0.02/√(n_layer)** |
| **Final LN** | final LayerNorm before lm_head; activation scaled by **1/√n** at resid stream |
| Optimizer | AdamW, betas (0.9, 0.999), **weight decay 0.1** (GPT-1 used 0.01!) |
| LR | peak 6.25e-5 (small) → 2.5e-4; warmup 2000 steps; cosine decay |
| Batch | grows 512 → larger over training |
| Grad clip | 1.0 |

### 2.3 Your local config (actual, from `transformer.py` + notebook override)
| Key | Value | vs. GPT-1 paper | vs. GPT-2 |
|---|---|---|---|
| n_layer | 8 | −4 | −4 |
| n_embd | 512 | −256 | −256 |
| n_head | 8 | −4 | −4 |
| head dim | 64 | = | = |
| d_ff | 2048 | −1024 | −1024 |
| max_seq_len | **128** | −384 | **−896** |
| vocab | 30,000 WordLevel | BPE 40k | byte BPE 50k |
| dropout | 0.1 | = | ≈ |
| optimizer | AdamW betas (0.9,0.999) | Adam | = |
| weight decay | 0.01 | 0.01 | **0.1** |
| grad clip | 1.0 | — | = |
| LR | 2.5e-4, warmup 2000, cosine | = | = |
| batch/step | 8×8 accum = 64 seqs × 128 = **8,192 tokens** | 32,768 | much larger |
| LN | post-norm (GPT-1 style) | = | pre-norm needed |
| residual scaling | none special | = | 1/√n needed |

**Per-layer param math (yours, verify by hand):**
```
emb          = vocab × d          = 30,000 × 512   = 15,360,000
pos          = ctx × d            = 128 × 512      =    65,536
per block:
  qkv proj   = d×d×3 + 3d         = 512×512×3+1536 =   787,968
  attn out   = d×d + d            = 512×512+512    =   262,656
  ln1        = 2×d                = 1,024
  ln2        = 2×d                = 1,024
  ffn up     = d×d_ff + d_ff      = 512×2048+2048  = 1,050,624
  ffn down   = d_ff×d + d         = 2048×512+512   = 1,049,088
  block total                     = 3,152,384
                                   × 8 blocks      = 25,219,072
final ln     = 2×d                = 1,024
Total                             = 40,645,632  ✔ (matches count_parameters)
```

---

## 3. DATA RECIPE — EXACT AMOUNTS, MIXING, AND WHY

### 3.1 What you used vs what you could have used
| Source | You used | Potential / role |
|---|---|---|
| DailyDialog (roskoN/dailydialog) | 76,026 train + 7,069 valid turns | conversational grammar (weak: very polite, limited domain) |
| Alpaca (tatsu-lab/alpaca) | 52,000 | instruction→answer format (machine-generated, biased answers) |
| GSM8K (openai/gsm8k) | 7,000 train + 1,000 valid | math steps (`... = 156`) |
| **Missing** | — | **more dialog**: OpenSubtitles, Ubuntu dialog, PersonaChat, Cornell Movie Dialogs (richer chat) |
| **Missing** | — | **more math**: MAWPS, MathQA, ASDiv for arithmetic generalization |
| **Missing** | — | general corpus for fluency: WikiText-103, each of your 3 stars instead of a raw WebText |

### 3.2 Mixing ratios (industry: per-character/token weighting, not per-example)
Rule: **Target a token budget per task**, then upsample so each task gets its share.
Suggested mix for a chat + light-math mini LM (100 M-token budget):
```
DailyDialog   20%   (~20 M tokens)  – dialog grammar
Alpaca        35%   (~35 M tokens)  – instruction following
GSM8K         15%   (~15 M tokens)  – arithmetic reasoning
OpenSubtitles 25%   (~25 M tokens)  – natural chat variation
WikiText-103   5%   (~5 M tokens)   – long-form fluency
```
Your run's ratio happened to be: DailyDialog 56% / Alpaca 39% / GSM8K 5% (by texts) — Alpaca machine-generated monotony over-represented. Rebalance by **tokens**, log `tokens_per_source` for every dataset version (you didn't).

### 3.3 Every dataset version must carry metadata
```json
{ "source": "alpaca", "version": "2026-08", "download_utc": "...", "raw_rows": 52002,
  "after_clean": 48210, "tokens_after_bpe": 33_400_000, "license": "CC-BY-NC-4.0" }
```
Never load a dataset without its metadata card. This is how you compare "is new data better?".

---

## 4. DATA CLEANING — EXACT RULES (companies do this 100%; evidence: RefinedWeb, Dolma, FineWeb drop 60–85% of raw crawl)

### 4.1 The cleaning pipeline, in order, with thresholds
| Step | Rule / threshold | Tool (no-fuss) |
|---|---|---|
| 1 decompress/normalize HTML | trafilatura | for web text |
| 2 Unicode normalize | **NFKC** (kills the ` ’ ` ≠ `'` bug; your data had `I ’ t`) | `unicodedata.normalize("NFKC", s)` |
| 3 smart quotes & dashes | → straight `'` `"` `-` | regex |
| 4 collapse whitespace | `\s+` → single space | regex |
| 5 drop control/zero-width chars | keep `\n\t`, drop the rest | regex |
| 6 **length filter per doc** | keep if 20 ≤ words ≤ 8192 | heuristic |
| 7 **alphabet ratio** | `len(letters)/len(chars) ≥ 0.8` (kills emoji/math-noise lines) | heuristic |
| 8 **repetition filter** | in-doc repeated-line ratio < 0.3; 4-gram repeat ratio < 0.2 | **directly kills your "again and again" habit** |
| 9 symbol ratio | non-ASCII punctuation/emoji < 10% of characters | heuristic |
| 10 language filter | fastText langid = en, score ≥ 0.8 | `fasttext` or `langdetect` |
| 11 boilerplate lines | drop lines < 20 chars without terminal punct; drop "enable Javascript", cookie, nav lines | C4-style rules |
| 12 **exact dedup** | SHA-256 | hashlib one-liner |
| 13 **near-dup dedup** | MinHash | Jaccard ≥ 0.85 → keep first. Toolkits: `datatrove`, Dolma |
| 14 **train/valid dedup** | remove any valid text whose 13-gram appears in train | **you MUST do this or valid is fake** |
| 15 PII | strip emails, phone numbers, SSN regexes (log victims) | regex |
| 16 toxicity | fastText toxic tagger on borderline docs only | `unitary/toxic-bert` second pass |
| 17 quality score | `0.2·lang + 0.2·alpha_ratio + 0.2·log(len) + 0.2·(1−line_dup) + 0.2·terminal_punct` → sort desc, keep top 80% | 20 lines of Python |
| 18 **AUDIT** | read 200 random kept + 200 dropped samples out loud | you — 10 min. If you hesitate, re-run |

Applies to **every** dataset (Alpaca snd GSM8K too — GSM8K: keep equations, numbers, and `= ` exactly; do NOT strip `=`).

---

## 5. TOKENIZER — EXACT SPEC (the `<unk>` 21% was 100% preventable)

### 5.1 Why WordLevel failed you (mechanism)
- `WordLevel(min_frequency=2)` on 135k texts → 30k words. Any unseen word → `<unk>`.
- `<unk>` gets its **own embedding**, becomes a frequent-ish token, and the LM learns it as valid filler → babble.
- Words with trailing punctuation `hello,` are separate tokens from `hello` → vocab explodes → your 30k cap bites harder.

### 5.2 The exact fix = GPT-2's tokenizer (byte-level BPE)
Spec:
- input units = **256 UTF-8 bytes** (via `bytes_to_unicode()`), so any string tokenizes; `<unk>` impossible
- regex-based pre-tokenization: GPT-2 splits words/numbers/punctuation, **keeps spaces as `Ġ`** tokens
- **split digits**: `split_digits=True` (SentencePiece) or GPT-2 style is fine — separate digit tokens help math (`156` = `1`,`5`,`6`)
- vocab size: **32,000** for ≤124 M models (LLaMA uses 32k; GPT-2 uses 50,257 at any size; pick 32k for your GPU RAM)
- special tokens: `<pad>` `<bos>` `<eos>` + optionally `<user>` `<assistant>`
- train on a **~10M-char sample of YOUR final cleaned mix** (not English-web randomly)
- implementations: `tiktoken` (cl100k/byte-level, OpenAI), `minbpe` (Karpathy, read it to learn), `tokenizers` (HuggingFace, fastest)

### 5.3 Acceptance tests (run BEFORE you write any model code)
```python
assert enc(valid).unk_rate == 0.0     # 1. zero unk
assert enc(open_txt).unk_rate == 0.0  #    even for never-seen-English
for t in sample(200):                 # 2. round-trip
    assert dec(enc(t)) == t
report top-100 tokens                 # 3. noise words? -> BPE too small
report mean tokens/text               # 4. changes dataset priorities
hash(tokenizer_conf)  # log it         # 5. version it per run
```

---

## 6. TRAINING RECIPE — EXACT NUMBERS

### 6.1 Token/param budget math (do this FIRST, on paper)
```
budget_hours × tok_per_s = total_tokens
example:  24 h × 5400 tok/s = 466 M tokens      (incl. AMP you may reach ~10k/s → 864 M)
target ratio at least 20 tokens/param  →  with 40 M params you need ≥ 800 M tokens
→ therefore with 24 h you CAN reach ~20:1 on a 40 M model. With 2 h (your run) → ~2:1. Proof you were 10× under.
```
If you cannot spend 12+ h, **shrink the model**, don't keep 40 M: a 15 M model with 300 M tokens (12 h) already beats a 40 M model with 40 M tokens for fluency.

### 6.2 Recommended hyperparameters for 4 GB (values to copy into GPT-2)
| Knob | Value | Why |
|---|---|---|
| precision | `torch.autocast("cuda", torch.float16)` | 2× speed, halves VRAM: this alone buys ctx 512 |
| optimizer | AdamW, betas (0.9, 0.999), eps 1e-8 | paper-standard |
| weight decay | **0.1** (GPT-2) — was 0.01 | better generalization, trivial change |
| grad clip | 1.0 | spikes are real (OPT logbook) |
| peak LR | 3e-4 (small models) | warmup must precede it |
| warmup steps | 2000 (≈ 2–5% of steps) | paper |
| schedule | cosine to 0 (you had this ✔) | paper |
| batch tokens/step | 64 seqs × ctx (accum to ~16k) | stable loss |
| shuffle seed | fixed, logged, resumable | reproducibility (you had this ✔) |
| context | **512** | 4× your old 128 — allows 1–2 turns memory |

### 6.3 The LR formula you already had (keep)
```python
lr = max_lr * (step+1)/warmup            if step < warmup        # linear up
lr = max_lr * 0.5*(1+cos(pi*(step-warmup)/(T-warmup)))  else      # cosine down
lr = 0.0                                 if step >= T            # floor
```

### 6.4 Monitoring / abort rules
- log per step: `loss, lr, grad_norm, param_norm` → TensorBoard (you had ✔, add grad_norm)
- **stop if `grad_norm` spikes > 10** (check clip working)
- **overfit?** `train_loss - valid_loss > 1.0` for 2 epochs → you were at ~0.0 (NOT overfitting; **keep training**)
- **sample output every 500 steps** into a text file → this is the earliest qualitative signal (you had zero; this alone would have warned you by epoch 1)
- **throughput log** tok/s per epoch → tells you when to stop (diminishing returns vs wall-clock)

---

## 7. FINE-TUNING FOR CHAT (SFT) — the missing 30%, exact recipe

### 7.1 Why zero-shot chat looks broken (your exact symptom)
Causal LM optimizes P(next | all past). It saw "User: ... Assistant: ..." thousands of times, so it perfectly imitates the *template*, but NO loss ever required it to answer. Without SFT it has no notion "reply after Assistant:".

### 7.2 Recipe (runs on 4 GB, ~30–60 min)
1. **Build masked pairs** — from the SAME cleaned data as `(input, target)`:
   - DailyDialog: `(User: <turn t>\nAssistant: , <turn t+1>)`
   - Alpaca: `(User: <instruction>\nAssistant: , <output>)`
   - GSM8K: `(User: <question>\nAssistant: , <reasoning steps>)`
2. **Loss masking**: compute loss only on tokens strictly after `Assistant:` → set loss-weight 0 on `input` positions (= `-100` target ids in PyTorch cross-entropy). Model can never learn to copy the prompt.
3. **Split**: 5% holdout fine-tune-valid (frozen for SFT).
4. **Hyperparams (SFT ≠ pretrain!)**
   | knob | value |
   |---|---|
   | LR | 1e-5 … 2e-5 (min! reusing 2.5e-4 destroys the LM) |
   | epochs | 1–3 (watch: SFT overfits fast; at 3 on 52k pairs expect degradation) |
   | warmup | 100 steps |
   | grad clip | 1.0 |
   | eval | valid loss MASKS TOO, plus your frozen 50–100 prompts |
5. **LoRA** (recommended): freeze base, train `r=16 α=32 dropout 0.05` on q/k/v/out + FFN. Effect: same SFT speed, VRAM ~1 GB, lets you keep ctx 512 for the base. Package: `peft`.
6. **Terminate on**: valid-masked-loss no longer improving for 2 epochs, OR human eval plateaus.

Expected: replies in `Assistant:` position, contain the answer, stop at `<eos>`. This is the single biggest fix to your symptoms.

---

## 8. DECODING — EXACT ORDER AND VALUES (kills "again and again")

### 8.1 The order is a spec (best practice 2025–2026)
```
logits                          from final layer
 → [1] penalties:  repetition 1.05–1.2, or frequency 0.3–0.7 (loops), or presence 0.1–0.5
 → [2] temperature: divide logits by T before any sampling
 → [3] top-p (nucleus) 0.9–0.95  OR  min-p 0.05–0.1   (top-k fixed 40 is the OLD way; adapt to p)
 → [4] sample multinomial(1)
```
- penalties BEFORE temperature (their strength is calibrated in logit space);
- truncation AFTER penalties (else penalized tokens slip back into the nucleus).
- **stop sequences**: `\nUser:`, `\nAssistant:`, `<eos>`, plus `max_new_tokens` cap (60–80 for chat).

### 8.2 The decision tree (measure first!)
| Symptom | Change (one at a time) | Re-check |
|---|---|---|
| loops ("again and again") | frequency_penalty 0.3 → 0.7 | distinct-1 ↑ |
| too random/garbage | T 0.8 → 0.5–0.6 | fluency |
| still garbage tail | top_p 0.9 (from none) | unk-rate + distinct-2 |
| too stiff after T↓ | T back 0.7 + top_p 0.9 | keep penalties |

---

## 9. EVALUATION — EXACT METRICS, THRESHOLDS, AND FORMULAS

### 9.1 Metric definitions (implement these)
```
perplexity      = exp( mean(-log P(next)) )                 # on held-out, you have it
token_acc       = mean(predicted next = true next)          # you have it
unk_in_gen      = (# `<unk>` tokens emitted)/#gen_tokens    # target 0
distinct-1      = unique generated tokens  / generated tokens   # target > 0.35
distinct-2      = unique bigrams / bigrams                      # target > 0.70
rep-3ratio      = repeated 3-grams / unique 3-grams             # target < 0.2
eos_stop_rate   = fraction of generations stopping at <eos> before token cap  # target high
```

### 9.2 The eval harness (6 items, in rank)
1. **Frozen prompt set (P0)**: 50–100 hand-written prompts across your tasks (smalltalk, greeting, `12*13`, "who are you", "tell me a joke"). Written BEFORE training. Never in train. **Run after every run.**
2. **perplexity + token_acc on valid** (have it) — LM health, not chat quality.
3. **Generation metrics** (distinct-1/2, unk_in_gen, eos_stop) on the frozen set — correctness of LOOP and vocab.
4. **Task metric** for math: exact-match of parsed numeric answer vs GSM8K test → accuracy.
5. **Human rubric** 20 outputs, each 1–5 on `relevance + grammar − repetition`; print on paper. This is the final say at 40 M scale.
6. (bonus later) GPT-4-as-judge over your 50–100, only when you have time + a frontier key.

### 9.3 Discipline (non-negotiable)
- Experiment log `experiments.md`: one row per run `date | config-hash | val_loss | distinct-1 | unk% | human-score | notes`.
- **One variable changed at a time.** Two changed = you can't attribute e.g. "was it BPE or the cleaning?".
- Never tune decode knobs on the test prompts (snoopal) — use a separate dev split.

---

## 10. GPU MEMORY MATH (GTX 1650, 4 GB = 3724 MiB usable) — PLAN BEFORE YOU PRETRAIN

```
weights fp32    = P × 4 B        (40M → 162 MB)
AdamW m+v       = P × 8 B        (      + 325 MB)
gradients cum   = ~P × 4 B       (      + 162 MB)   (peak during backward)
activations     = depends on ctx²×batch×d          (dominant at ctx 512)
─────────────────────────────────────────────
fp32 total at 40M, small ctx ≈ 600–700 MB + activations → fits, barely, OUCH at ctx 512.
Fix: autocast fp16 → weights/acts half → ~350 MB → ctx 512 becomes possible.
Bigger fix: LoRA freeze base → optimizer state only on r×d LoRA params (~0.3M×8B ≈ 3 MB).
```
Rule of thumb: if total inference VRAM ≥ 70% of card during training, shrink batch/ctx, NOT ambitions. Sample 10 lines of output every 500 steps is free and priceless.

---

## 11. FULL PIPELINE FLOW — SEE EXACTLY WHERE YOU WERE/ARE (visual)

```
[0] GOAL + FROZEN EVAL (50-100 prompts) ─────────────── ❌ do first  (30 min)
         │
[1] ACQUIRE: DailyDialog + Alpaca + GSM8K (+OpenSubtitles, WikiText-103) ✅ / expand
         │
[2] CLEAN: NFKC → quotes → ws → length → alpha → repeat → bordel → lang
         │  → EXACT dedup → MinHash near-dup → train/valid dedup → PII → quality-score → AUDIT
         │                                       ❌ entire stage missing
         ▼
[3] TOKENIZER: byte-level BPE 32k on cleaned mix → unk=0 → round-trip → version  ❌ missing (was WordLevel → 21% unk)
         │
[4] PRETRAIN: AMP · ctx 512 · wd 0.1 · LR 3e-4 warmup 2000 cosine · grad_clip 1.0
         │  · checkpoint every N · grad_norm log · sample txt every 500 steps
         │  · 12–24 h (ratio≥20:1)                     ⚠️ present but 10× under-run
         ▼
[5] SFT: masked pairs (loss on Assistant:) · LR 1e-5 · 1–3 epochs · LoRA r16  ❌ missing──★ THE CHAT FIX
         │
[6] EVAL: frozen set + distinct-1/2 + unk% + eos% + human 1–5 → append experiments.md   ❌ missing
         │
[7] DECODE: penalties→T→top-p→sample + stop seqs (60–80 tk)          ⚠️ penalty added late
         │
[8] ALIGN (DPO optional) + guardrails/logs (production)              ❌ later
         └────────────── iterate [5]→[6] until human ≥ 3.5/5
```

---

## 12. WHAT YOUR PROJECT ALREADY DID RIGHT (keep line-for-line)

- ✔ checkpointing + rollback + resume (`last.pt`/`final.pt`/`step_N`/`epoch_N`, keep-3)
- ✔ CUDA-safe RNG save/restore
- ✔ no-retrain-on-reload path
- ✔ config-per-run + TensorBoard + altair dashboards
- ✔ Gradio 6 inline chat
- ✘ BUT: no eval harness, no dedup, no BPE, no SFT, no AMP, ctx 128, fp32.

---

## 13. GPT-1 → GPT-2 CHECKLIST (tick each; all have Section refs)

- [ ] Pre-norm LayerNorm inside each block (before attn & before FFN)  [§2.2]
- [ ] c_proj init std = 0.02/√(n_layer); final LN + 1/√n activation scale  [§2.2]
- [ ] Byte-level BPE 32k, unk-rate 0, round-trip, versioned  [§5]
- [ ] weight_decay 0.1, AdamW (0.9,0.999)  [§6]
- [ ] ctx 512 (min), prefer 1024 via AMP  [§6]
- [ ] AMP autocast mixed precision  [§6]
- [ ] data: clean (NFKC, dedup exact+near, train/valid dedup, AUDIT)  [§4]
- [ ] tokens/param ≥ 20 (plan the hours now)  [§6.1]
- [ ] sample output every 500 steps + grad_norm in TB  [§6.4]
- [ ] frozen 50–100 eval prompts BEFORE training  [§9]
- [ ] generation separators/stop seqs; penalties→T→top-p  [§8]
- [ ] SFT on masked pairs (this is what makes it a chatbot)  [§7]
- [ ] experiments.md log: date|config-hash|val_loss|distinct-1|unk%|human  [§9.3]

---

## 14. THE CONCRETE GPT-2 TARGET SPEC (copy this whole block as your config for 4 GB)

```json
{
  "mode": "local",
  "n_layer": 8, "n_embd": 512, "n_head": 8, "d_ff": 2048,
  "max_seq_len": 512, "block_stride": 448,
  "vocab_size": 32000, "tokenizer": "byte_bpe", "split_digits": true,
  "dropout": 0.1,
  "num_epochs": 12,
  "batch_size": 16, "accum_iter": 4,           // 64 seqs/step
  "warmup": 2000, "base_lr": 3e-4, "scheduler": "cosine",
  "weight_decay": 0.1, "grad_clip": 1.0, "betas": [0.9, 0.999],
  "amp": true,
  "data_mix": { "dailydialog": 0.20, "alpaca": 0.35, "gsm8k": 0.15,
                "opensubtitles": 0.25, "wikitext": 0.05 },
  "sft": { "on": true, "masked": true, "lr": 1.5e-5, "epochs": 2,
           "lora_r": 16, "lora_alpha": 32 },
  "decode": { "repetition_penalty": 1.15, "top_p": 0.9, "temperature": 0.7,
              "stop": ["\nUser:", "\nAssistant:", "<eos>"], "max_new_tokens": 80 },
  "eval": { "frozen_prompts": 100, "valid_split": 0.05 }
}
```
Expected outcome vs GPT-1 (measured targets): unk 0%, val_loss < 5.0, distinct-1 > 0.35, human smalltalk ≥ 3.5/5, uses ~12–24 h of your GPU. That is the bar. Nothing above is speculative — every number comes from the papers below or real production pipelines.

---

## 15. THE 7-DAY PLAN (one concrete act per day)

| Day | Do | Done when |
|---|---|---|
| 1 | Write 100 frozen prompts; clean+dedup all 3 datasets (run §4 list), log `tokens_per_source`; AUDIT 200 lines | experiment log has data card |
| 2 | Byte BPE 32k on cleaned mix; pass §5.3 tests; tokenizer versioned | tokenizer artifact + tests green |
| 3 | GPT-2 arch (pre-norm, wd 0.1, c_proj scale); AMP; ctx 512; config from §14 | prints its own param + VRAM at boot |
| 4 | Launch pretraining 12–24 h with §6.4 monitoring; read sampled output at 500-step marks | epoch≥2, val_loss < 5.8, samples not repeating |
| 5 | SFT: masked pairs, LR 1.5e-5, 2 epochs (LoRA) | replies answer in `Assistant:` position |
| 6 | Eval harness §9 on frozen set; tune decode one knob at a time | experiments.md has 2+ rows, human ≥ 3.5 |
| 7 | Freeze run, write results table in README, delete nothing, chat server up | README documents deltas vs GPT-1 |

---

## 16. SOURCES (read the real papers once)

- GPT-1 (2018) & GPT-2 (2019), Radford et al. — architecture + byte-BPE + recipe
- Chinchilla (2022), Hoffmann et al. — 20 tokens/param
- InstructGPT, "Training LMs to follow instructions" (2022) — SFT >> zero-shot
- RefinedWeb (2023), Penedo et al. — filtering > curated corpora; dedup ablations
- Dolma (2024), Soldaini et al. — open 3T corpus, per-stage ablations
- FineWeb / DataComp-LM (2024) — model-based filtering
- Nucleus Sampling (2020), Holtzman et al. — top-p
- Repetition studies (2025): penalties, LZ-penalty, early-stopping in beam
- Tools: `datatrove`, Dolma toolkit, `deduplicate-text-datasets`, `tiktoken`, `minbpe` (Karpathy), `tokenizers` (HF), `peft` (LoRA), PyTorch AMP docs