# -*- coding: utf-8 -*-
"""gpt-v1: Decoder-only language model following GPT-1 paper ideas.

Paper: Radford et al., "Improving Language Understanding by Generative Pre-Training" (2018)

Paper architecture (full GPT-1):
  - Decoder-only Transformer (causal / masked self-attention)
  - L=12 layers, d_model=768, heads=12, d_ff=3072
  - Context window = 512 tokens
  - Learned token + position embeddings
  - GELU activation, dropout 0.1
  - Weight init N(0, 0.02)
  - Adam, max LR 2.5e-4, 2000-step warmup, cosine decay
  - Objective: next-token language modeling (zero-shot generation after pretrain)

This codebase:
  - Same architecture family as the paper
  - `local` mode: smaller size so a 4GB laptop GPU can train
  - `kaggle` / `full` modes: closer to paper scale (12/768/12)
  - Datasets from Hugging Face: DailyDialog + Alpaca + GSM8K
  - All knobs overrideable via get_run_config(override) / run_pipeline(override)
"""

from __future__ import annotations

import glob
import math
import os
import random
import re
import time
from os.path import exists

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def running_on_kaggle() -> bool:
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.path.isdir("/kaggle/working"))


def default_run_mode() -> str:
    env = os.environ.get("RUN_MODE")
    if env:
        return env.lower().strip()
    return "kaggle" if running_on_kaggle() else "local"


RUN_MODE = default_run_mode()
if running_on_kaggle() and os.path.isdir("/kaggle/working"):
    os.chdir("/kaggle/working")


def login_huggingface() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("No HF_TOKEN set; anonymous HF downloads (rate limits possible).", flush=True)
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        print("Hugging Face login: OK", flush=True)
    except Exception as err:
        print("Hugging Face login skipped:", err, flush=True)


login_huggingface()


# ---------------------------------------------------------------------------
# Config (paper defaults + hardware profiles)
# ---------------------------------------------------------------------------

# Paper GPT-1 reference values (for comments / full mode target)
PAPER_GPT1 = {
    "n_layer": 12,
    "n_embd": 768,
    "n_head": 12,
    "d_ff": 3072,
    "max_seq_len": 512,
    "base_lr": 2.5e-4,
    "warmup": 2000,
    "dropout": 0.1,
    "weight_decay": 0.01,
}


def base_config(mode: str) -> dict:
    """Return a complete run config. Modes: local | full | kaggle."""
    # Shared defaults inspired by the paper training recipe
    common = {
        "mode": mode,
        "num_epochs": 5,  # paper pretrained ~100 epochs on BooksCorpus; we use 5 for practical runs
        "base_lr": PAPER_GPT1["base_lr"],
        "warmup": PAPER_GPT1["warmup"],
        "weight_decay": PAPER_GPT1["weight_decay"],
        "grad_clip": 1.0,
        "dropout": PAPER_GPT1["dropout"],
        "save_every_steps": 1200,
        "data_seed": 42,
        # HF datasets for chat + instructions + math reasoning
        "train_sources": ["dailydialog", "alpaca", "gsm8k"],
        "valid_sources": ["dailydialog", "gsm8k"],
        "train_pairs": None,
        "valid_pairs": None,
        "alpaca_train_pairs": 52000,
        "gsm8k_train_pairs": 7000,
        "gsm8k_valid_pairs": 1000,
        "betas": (0.9, 0.999),  # Adam betas from paper
    }

    if mode == "local":
        # Laptop (e.g. GTX 1650 4GB): same architecture family, reduced width/depth/context
        common.update(
            {
                "batch_size": 8,
                "accum_iter": 8,  # effective batch ~64 tokens-blocks
                "max_seq_len": 128,  # paper=512; reduced for VRAM
                "block_stride": 96,
                "n_layer": 8,  # paper=12
                "n_head": 8,  # paper=12
                "n_embd": 512,  # paper=768
                "d_ff": 2048,  # paper=3072 (4 * d_model)
                "vocab_path": "gptv1_vocab_local.json",
                "file_prefix": "gptv1_local_",
            }
        )
        return common

    if mode == "kaggle":
        # Closer to paper GPT-1 size on 2xT4
        common.update(
            {
                "batch_size": 16,
                "accum_iter": 4,
                "max_seq_len": 256,  # paper=512; still reduced for memory
                "block_stride": 192,
                "n_layer": 12,
                "n_head": 12,
                "n_embd": 768,
                "d_ff": 3072,
                "warmup": 2000,
                "alpaca_train_pairs": 300000,
                "gsm8k_train_pairs": 7500,
                "gsm8k_valid_pairs": 1200,
                "vocab_path": "gptv1_vocab_kaggle.json",
                "file_prefix": "gptv1_kaggle_",
                "save_every_steps": 1500,
            }
        )
        return common

    # full: paper-like architecture on a stronger single GPU
    common.update(
        {
            "batch_size": 12,
            "accum_iter": 8,
            "max_seq_len": 256,
            "block_stride": 192,
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "d_ff": 3072,
            "warmup": 2000,
            "alpaca_train_pairs": 100000,
            "gsm8k_train_pairs": 7000,
            "gsm8k_valid_pairs": 1000,
            "vocab_path": "gptv1_vocab_full.json",
            "file_prefix": "gptv1_full_",
            "save_every_steps": 1500,
        }
    )
    return common


def get_run_config(override=None) -> dict:
    mode = str(RUN_MODE).lower()
    if mode not in ("local", "full", "kaggle"):
        raise ValueError("RUN_MODE must be one of: local, full, kaggle")
    cfg = base_config(mode)
    if override:
        cfg = dict(cfg)
        cfg.update(override)
    # Keep d_ff consistent with paper rule if user changes n_embd but not d_ff
    if "d_ff" not in (override or {}) and "n_embd" in (override or {}):
        cfg["d_ff"] = 4 * int(cfg["n_embd"])
    return cfg


# ---------------------------------------------------------------------------
# Data (Hugging Face)
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text))
    text = re.sub(r"\s+'\s+", "'", text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    return text.strip()


def build_dailydialog_texts(split="train", max_pairs=None):
    raw = load_dataset(
        "roskoN/dailydialog",
        revision="refs/convert/parquet",
        token=os.environ.get("HF_TOKEN"),
    )
    texts = []
    for entry in raw[split]:
        utts = [normalize_text(u) for u in entry["utterances"] if normalize_text(u)]
        for i in range(len(utts) - 1):
            texts.append(f"User: {utts[i]}\nAssistant: {utts[i + 1]}")
            if max_pairs is not None and len(texts) >= max_pairs:
                return texts
    return texts


def build_alpaca_texts(max_pairs=None):
    ds = load_dataset("tatsu-lab/alpaca", token=os.environ.get("HF_TOKEN"))["train"]
    texts = []
    for row in ds:
        inst = normalize_text(row.get("instruction", ""))
        inp = normalize_text(row.get("input", ""))
        out = normalize_text(row.get("output", ""))
        if not out:
            continue
        prompt = inst if not inp else f"{inst}\n{inp}"
        if prompt:
            texts.append(f"User: {prompt}\nAssistant: {out}")
        if max_pairs is not None and len(texts) >= max_pairs:
            break
    return texts


def build_gsm8k_texts(split="train", max_pairs=None):
    ds = load_dataset(
        "openai/gsm8k",
        "default",
        revision="refs/convert/parquet",
        token=os.environ.get("HF_TOKEN"),
    )[split]
    texts = []
    for row in ds:
        q = normalize_text(row.get("question", ""))
        a = normalize_text(row.get("answer", ""))
        if not q or not a:
            continue
        texts.append(f"User: Solve this step by step. {q}\nAssistant: {a}")
        if max_pairs is not None and len(texts) >= max_pairs:
            break
    return texts


def build_train_valid_texts(config):
    train_sources = list(config.get("train_sources") or ["dailydialog"])
    valid_sources = list(config.get("valid_sources") or ["dailydialog"])
    train_texts, valid_texts = [], []
    train_cap = config.get("train_pairs")
    valid_cap = config.get("valid_pairs")

    if "dailydialog" in train_sources:
        train_texts.extend(build_dailydialog_texts("train", max_pairs=train_cap))
    if "alpaca" in train_sources:
        train_texts.extend(build_alpaca_texts(max_pairs=int(config.get("alpaca_train_pairs") or 0)))
    if "gsm8k" in train_sources:
        train_texts.extend(build_gsm8k_texts("train", max_pairs=int(config.get("gsm8k_train_pairs") or 0)))

    if "dailydialog" in valid_sources:
        valid_texts.extend(build_dailydialog_texts("validation", max_pairs=valid_cap))
    if "gsm8k" in valid_sources:
        valid_texts.extend(build_gsm8k_texts("test", max_pairs=int(config.get("gsm8k_valid_pairs") or 0)))

    rng = random.Random(int(config.get("data_seed", 42)))
    rng.shuffle(train_texts)
    rng.shuffle(valid_texts)
    if train_cap is not None:
        train_texts = train_texts[: int(train_cap)]
    if valid_cap is not None:
        valid_texts = valid_texts[: int(valid_cap)]

    print(
        "Data sources -> train: %s | valid: %s"
        % (",".join(train_sources), ",".join(valid_sources)),
        flush=True,
    )
    return train_texts, valid_texts


def build_vocab(texts, vocab_path):
    if exists(vocab_path):
        tok = Tokenizer.from_file(vocab_path)
        print(f"Loaded vocab {vocab_path} size={tok.get_vocab_size()}", flush=True)
        return tok

    # Paper used BPE (~40k). WordLevel is simpler for learning; upgrade later if needed.
    print("Building WordLevel vocabulary (paper used BPE ~40k)...", flush=True)
    tok = Tokenizer(WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = WhitespaceSplit()
    trainer = WordLevelTrainer(
        min_frequency=2,
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(vocab_path)
    print(f"Saved vocab {vocab_path} size={tok.get_vocab_size()}", flush=True)
    return tok


def encode_texts(tokenizer, texts):
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    ids = []
    for t in texts:
        ids.extend([bos] + tokenizer.encode(t).ids + [eos])
    return ids


class LMBlocksDataset(Dataset):
    """Contiguous token blocks for next-token LM (paper style)."""

    def __init__(self, token_ids, max_seq_len=128, stride=96):
        self.x, self.y = [], []
        if len(token_ids) < max_seq_len + 1:
            return
        for i in range(0, len(token_ids) - max_seq_len - 1, stride):
            chunk = token_ids[i : i + max_seq_len + 1]
            self.x.append(torch.tensor(chunk[:-1], dtype=torch.long))
            self.y.append(torch.tensor(chunk[1:], dtype=torch.long))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model (GPT-1 style decoder-only Transformer)
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """Masked multi-head self-attention (paper: 12 heads, d_model=768)."""

    def __init__(self, n_embd, n_head, dropout, max_seq_len):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise FFN: d_model -> 4*d_model -> d_model with GELU (paper)."""

    def __init__(self, n_embd, d_ff, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, d_ff)
        self.c_proj = nn.Linear(d_ff, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """Transformer decoder block (post-LN style used in original Transformer / GPT-1 era)."""

    def __init__(self, n_embd, n_head, d_ff, dropout, max_seq_len):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, dropout, max_seq_len)
        self.mlp = MLP(n_embd, d_ff, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.mlp(x))
        return x


class GPTDecoderOnly(nn.Module):
    """Decoder-only LM matching GPT-1 structural choices."""

    def __init__(self, vocab_size, max_seq_len, n_layer, n_head, n_embd, d_ff, dropout):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.n_embd = n_embd
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_seq_len, n_embd)  # learned positions (paper)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, d_ff, dropout, max_seq_len) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Paper: N(0, 0.02)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        if T > self.max_seq_len:
            raise ValueError(f"sequence length {T} > max_seq_len {self.max_seq_len}")
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=80, temperature=0.8, top_k=40, repetition_penalty=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if repetition_penalty > 1.0:
                for s in range(idx.shape[0]):
                    seen = idx[s].tolist()
                    for t in set(seen):
                        if logits[s, t] > 0:
                            logits[s, t] /= repetition_penalty
                        else:
                            logits[s, t] *= repetition_penalty
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def make_model(vocab_size, config) -> GPTDecoderOnly:
    return GPTDecoderOnly(
        vocab_size=vocab_size,
        max_seq_len=int(config["max_seq_len"]),
        n_layer=int(config["n_layer"]),
        n_head=int(config["n_head"]),
        n_embd=int(config["n_embd"]),
        d_ff=int(config.get("d_ff", 4 * int(config["n_embd"]))),
        dropout=float(config["dropout"]),
    )


def print_architecture(config, vocab_size=None):
    print("=" * 72, flush=True)
    print("GPT-v1 architecture (paper-inspired)", flush=True)
    print("Paper GPT-1: L=12, d=768, heads=12, d_ff=3072, ctx=512", flush=True)
    print(
        "This run: L=%d, d=%d, heads=%d, d_ff=%d, ctx=%d"
        % (
            config["n_layer"],
            config["n_embd"],
            config["n_head"],
            config.get("d_ff", 4 * config["n_embd"]),
            config["max_seq_len"],
        ),
        flush=True,
    )
    if vocab_size is not None:
        tmp = make_model(vocab_size, config)
        print(f"Approx trainable params: {count_parameters(tmp):,}", flush=True)
        del tmp
    print("=" * 72, flush=True)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def checkpoint_paths(config):
    p = config["file_prefix"]
    return {
        "last": f"{p}last.pt",
        "final": f"{p}final.pt",
        "epoch": lambda e: f"{p}epoch_{e}.pt",
        "step": lambda s: f"{p}step_{s}.pt",
    }


def collect_rng_state():
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    cpu = state["torch"]
    if not isinstance(cpu, torch.Tensor):
        cpu = torch.tensor(cpu, dtype=torch.uint8)
    if cpu.dtype != torch.uint8:
        cpu = cpu.to(torch.uint8)
    if cpu.device.type != "cpu":
        cpu = cpu.cpu()
    torch.set_rng_state(cpu)
    if torch.cuda.is_available() and "cuda" in state:
        saved = list(state["cuda"])[: torch.cuda.device_count()]
        cuda_states = [s.cpu().to(torch.uint8) for s in saved]
        torch.cuda.set_rng_state_all(cuda_states)


def save_checkpoint(path, payload):
    torch.save(payload, path)
    print(f"Saved {path}", flush=True)


def cleanup_old_checkpoints(prefix, keep=3):
    for pattern in (f"{prefix}step_*.pt", f"{prefix}epoch_*.pt"):
        files = sorted(glob.glob(pattern), key=os.path.getmtime)
        while len(files) > keep:
            old = files.pop(0)
            try:
                os.remove(old)
                print(f"Deleted old checkpoint: {old}", flush=True)
            except FileNotFoundError:
                pass


def build_checkpoint(model, optimizer, scheduler, epoch, step, config, val_loss):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "step": int(step),
        "config": dict(config),
        "val_loss": float(val_loss) if val_loss is not None else None,
        "rng": collect_rng_state(),
    }


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def cosine_lr(step, max_lr, warmup, total_steps):
    """Paper: linear warmup then cosine decay to 0."""
    if step < warmup:
        return max_lr * float(step + 1) / float(max(1, warmup))
    if step >= total_steps:
        return 0.0
    progress = float(step - warmup) / float(max(1, total_steps - warmup))
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate_loss(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            _, loss = model(xb, yb)
            losses.append(float(loss.item()))
    return float("inf") if not losses else sum(losses) / len(losses)


def create_dataloaders_for_lm(tokenizer, train_texts, valid_texts, config):
    train_ids = encode_texts(tokenizer, train_texts)
    valid_ids = encode_texts(tokenizer, valid_texts)
    train_ds = LMBlocksDataset(train_ids, int(config["max_seq_len"]), int(config["block_stride"]))
    valid_ds = LMBlocksDataset(valid_ids, int(config["max_seq_len"]), int(config["block_stride"]))
    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=int(config["batch_size"]), shuffle=False)
    return train_loader, valid_loader


def train_worker(tokenizer, train_loader, valid_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}", flush=True)
    print_architecture(config, vocab_size=tokenizer.get_vocab_size())

    model = make_model(tokenizer.get_vocab_size(), config).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)
    module = model.module if hasattr(model, "module") else model
    print(f"Trainable params: {count_parameters(module):,}", flush=True)

    betas = tuple(config.get("betas", (0.9, 0.999)))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["base_lr"]),
        betas=betas,
        weight_decay=float(config["weight_decay"]),
    )

    paths = checkpoint_paths(config)
    save_every = int(config.get("save_every_steps", 1000))
    start_epoch = 0
    global_step = 0

    if exists(paths["last"]):
        print(f"Resuming from {paths['last']}", flush=True)
        ckpt = torch.load(paths["last"], map_location=device)
        module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("step", 0))
        restore_rng_state(ckpt.get("rng"))

    accum = int(config["accum_iter"])
    grad_clip = float(config.get("grad_clip", 0.0))
    num_epochs = int(config["num_epochs"])
    steps_per_epoch = max(1, len(train_loader) // max(1, accum))
    total_steps = max(1, steps_per_epoch * num_epochs)
    warmup = int(config["warmup"])
    max_lr = float(config["base_lr"])

    for epoch in range(start_epoch, num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        seen = 0
        t0 = time.time()

        for i, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)
            _, loss = model(xb, yb)
            (loss / accum).backward()

            if (i + 1) % accum == 0:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                lr = cosine_lr(global_step, max_lr, warmup, total_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if save_every > 0 and global_step % save_every == 0:
                    val_mid = evaluate_loss(model, valid_loader, device)
                    payload = build_checkpoint(module, optimizer, None, epoch, global_step, config, val_mid)
                    save_checkpoint(paths["step"](global_step), payload)
                    save_checkpoint(paths["last"], payload)
                    cleanup_old_checkpoints(config["file_prefix"], keep=3)

            total_loss += float(loss.item())
            seen += 1
            if i % 80 == 1:
                elapsed = max(1e-6, time.time() - t0)
                tok_s = xb.numel() * (i + 1) / elapsed
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch} | Batch {i} | Loss {total_loss / max(1, seen):.4f} | "
                    f"Speed {tok_s:.0f} tok/s | LR {lr_now:.8f}",
                    flush=True,
                )

        val_loss = evaluate_loss(model, valid_loader, device)
        val_ppl = math.exp(min(20.0, val_loss))
        train_loss = total_loss / max(1, seen)
        print("-" * 72, flush=True)
        print(
            f"Epoch {epoch + 1}/{num_epochs} done | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}",
            flush=True,
        )
        print("-" * 72, flush=True)

        payload = build_checkpoint(module, optimizer, None, epoch, global_step, config, val_loss)
        save_checkpoint(paths["epoch"](epoch), payload)
        save_checkpoint(paths["last"], payload)
        cleanup_old_checkpoints(config["file_prefix"], keep=3)

    final_val = evaluate_loss(model, valid_loader, device)
    payload = build_checkpoint(module, optimizer, None, num_epochs - 1, global_step, config, final_val)
    save_checkpoint(paths["final"], payload)
    save_checkpoint(paths["last"], payload)
    print("Training complete.", flush=True)
    return module


def load_or_train_model(tokenizer, train_loader, valid_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(tokenizer.get_vocab_size(), config)
    paths = checkpoint_paths(config)

    if exists(paths["final"]):
        ckpt_path = paths["final"]
        print(f"Loading final checkpoint: {ckpt_path}", flush=True)
    elif exists(paths["last"]):
        print("No final checkpoint. Resume training from last.", flush=True)
        train_worker(tokenizer, train_loader, valid_loader, config)
        ckpt_path = paths["final"] if exists(paths["final"]) else paths["last"]
    else:
        print("No checkpoint found. Starting training.", flush=True)
        train_worker(tokenizer, train_loader, valid_loader, config)
        ckpt_path = paths["final"] if exists(paths["final"]) else paths["last"]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Generation / chat
# ---------------------------------------------------------------------------

def decode_ids(ids, tokenizer):
    eos = tokenizer.token_to_id("<eos>")
    pad = tokenizer.token_to_id("<pad>")
    bos = tokenizer.token_to_id("<bos>")
    toks = []
    for t in ids:
        i = int(t)
        if i in (pad, bos):
            continue
        if i == eos:
            break
        toks.append(tokenizer.id_to_token(i))
    return " ".join(toks).strip()


def generate_reply(model, tokenizer, prompt, max_new_tokens=80, temperature=0.8, top_k=40, repetition_penalty=1.1):
    device = next(model.parameters()).device
    bos = tokenizer.token_to_id("<bos>")
    prompt_ids = tokenizer.encode(f"User: {normalize_text(prompt)}\nAssistant:").ids
    x = torch.tensor([[bos] + prompt_ids], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, repetition_penalty=repetition_penalty)[0]
    generated = y[len(x[0]) :]  # keep only the newly generated tokens (no prompt echo)
    return decode_ids(generated.tolist(), tokenizer)


def launch_gradio_chat(model, tokenizer):
    try:
        import gradio as gr
    except ImportError:
        import subprocess
        import sys

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio"])
        import gradio as gr

    def submit(msg, history):
        history = list(history or [])
        text = (msg or "").strip()
        if not text:
            return history, ""
        ans = generate_reply(model, tokenizer, text)
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": ans})
        return history, ""

    with gr.Blocks(title="gpt-v1 decoder chat") as demo:
        gr.Markdown("## GPT-v1 Decoder-only Chat (zero-shot after LM pretraining)")
        chatbot = gr.Chatbot(height=420)
        txt = gr.Textbox(lines=2, placeholder="Type a message...")
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")
        send.click(submit, [txt, chatbot], [chatbot, txt])
        txt.submit(submit, [txt, chatbot], [chatbot, txt])
        clear.click(lambda: ([], ""), outputs=[chatbot, txt])
    return demo.launch(inline=True, quiet=False)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(config_override=None):
    cfg = get_run_config(config_override)

    print("=" * 72, flush=True)
    print("GPT-v1 Decoder-only pipeline (paper-inspired)", flush=True)
    print("Config:", cfg, flush=True)
    print("=" * 72, flush=True)

    print("Step 1/6: Load HF datasets...", flush=True)
    train_texts, valid_texts = build_train_valid_texts(cfg)
    print(f"Train texts: {len(train_texts)} | Valid texts: {len(valid_texts)}", flush=True)

    print("Step 2/6: Vocabulary...", flush=True)
    tokenizer = build_vocab(train_texts, cfg["vocab_path"])

    print("Step 3/6: LM blocks / dataloaders...", flush=True)
    train_loader, valid_loader = create_dataloaders_for_lm(tokenizer, train_texts, valid_texts, cfg)
    print(f"Train batches: {len(train_loader)} | Valid batches: {len(valid_loader)}", flush=True)

    print("Step 4/6: Train or load model...", flush=True)
    model = load_or_train_model(tokenizer, train_loader, valid_loader, cfg)

    print("Step 5/6: Final evaluation...", flush=True)
    device = next(model.parameters()).device
    val_loss = evaluate_loss(model, valid_loader, device)
    val_ppl = math.exp(min(20.0, val_loss))
    print(f"Validation loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}", flush=True)

    print("Step 6/6: Zero-shot samples...", flush=True)
    for q in ["hello", "solve 12*13", "write a short thank-you message"]:
        print("User:", q, flush=True)
        print("GPT-v1:", generate_reply(model, tokenizer, q), flush=True)
        print("-" * 50, flush=True)

    return model, tokenizer, cfg


if __name__ == "__main__":
    run_pipeline()
