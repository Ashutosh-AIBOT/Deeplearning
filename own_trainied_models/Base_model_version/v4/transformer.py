# -*- coding: utf-8 -*-
"""v3: Annotated Transformer + DailyDialog with resume checkpoints and chat.

Same paper model as v2. Adds:
  - native PyTorch checkpoint dicts (model/optimizer/scheduler/epoch/step)
  - resume from `{prefix}last.pt` after interrupt
  - greedy-decode chat loop (one-turn replies, not ChatGPT)

Dataset: Hugging Face `roskoN/dailydialog` (not a pretrained HF model).
Use conda env `transformer-env`.
"""

import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))

import re
import os
import random
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
import altair as alt
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import spacy
import GPUtil
import warnings

warnings.filterwarnings("ignore")
RUN_EXAMPLES = True

# Hugging Face token for faster/authenticated dataset downloads.
# Keep this notebook/file private. If it was shared, revoke the token on huggingface.co.
os.environ["HF_TOKEN"] = os.environ.get(
    "HF_TOKEN", "hf_YOUR_HUGGINGFACE_TOKEN"
)
os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]


def login_huggingface():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("No HF_TOKEN set; dataset download will be anonymous.", flush=True)
        return
    try:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
        print("Hugging Face login: OK", flush=True)
    except Exception as err:
        print("Hugging Face login skipped:", err, flush=True)


login_huggingface()


def setup_human_numbers():
    """Show normal decimals everywhere (0.000011), not 1.1e-05."""

    def _fmt(v):
        try:
            x = float(v)
        except Exception:
            return str(v)
        if abs(x) >= 1:
            s = "%.4f" % x
        else:
            s = "%.8f" % x
        return s.rstrip("0").rstrip(".")

    pd.set_option("display.float_format", _fmt)
    pd.set_option("display.max_colwidth", 100)
    pd.set_option("display.width", 120)
    try:
        import numpy as np

        np.set_printoptions(suppress=True, precision=4)
    except Exception:
        pass
    print("Number display: normal decimals (not scientific like 1.1e-05)", flush=True)


setup_human_numbers()


def running_on_kaggle():
    return bool(
        os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
        or os.path.isdir("/kaggle/working")
    )


def default_run_mode():
    env = os.environ.get("RUN_MODE")
    if env:
        return env
    # Kaggle: use the v4 research profile unless overridden.
    if running_on_kaggle():
        return "kaggle_v4"
    return "local"


# "local" = laptop one-go; "full" = Colab/Kaggle
RUN_MODE = default_run_mode()
if running_on_kaggle() and os.path.isdir("/kaggle/working"):
    os.chdir("/kaggle/working")
    print("Kaggle detected. cwd=/kaggle/working  RUN_MODE=%s" % RUN_MODE)

# Some convenience helper functions used throughout the notebook


def is_interactive_notebook():
    return __name__ == "__main__"


def show_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        return fn(*args)


def execute_example(fn, args=[]):
    if __name__ == "__main__" and RUN_EXAMPLES:
        fn(*args)


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))



class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def example_mask():
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Subsequent Mask": subsequent_mask(20)[0][x, y].flatten(),
                    "Window": y,
                    "Masking": x,
                }
            )
            for y in range(20)
            for x in range(20)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect()
        .properties(height=250, width=250)
        .encode(
            alt.X("Window:O"),
            alt.Y("Masking:O"),
            alt.Color("Subsequent Mask:Q", scale=alt.Scale(scheme="viridis")),
        )
        .interactive()
    )


show_example(example_mask)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

def example_positional():
    pe = PositionalEncoding(20, 0)
    y = pe.forward(torch.zeros(1, 100, 20))

    data = pd.concat(
        [
            pd.DataFrame(
                {
                    "embedding": y[0, :, dim],
                    "dimension": dim,
                    "position": list(range(100)),
                }
            )
            for dim in [4, 5, 6, 7]
        ]
    )

    return (
        alt.Chart(data)
        .mark_line()
        .properties(width=800)
        .encode(x="position", y="embedding", color="dimension:N")
        .interactive()
    )


show_example(example_positional)

def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def inference_test():
    test_model = make_model(11, 11, 2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


show_example(run_tests)

"""# Part 2: Model Training"""

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    def __init__(self):
        self.step = 0
        self.accum_step = 0
        self.samples = 0
        self.tokens = 0


def human_lr(lr):
    """Write learning rate as a normal decimal (0.000011), not 1.1e-05."""
    return ("%.8f" % float(lr)).rstrip("0").rstrip(".")


# Commented out IPython magic to ensure Python compatibility.
def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
    on_step=None,
    history=None,
    epoch=0,
):
    """Train a single epoch.

    Training math is unchanged from the paper notebook.
    Optional `history` only records loss/lr for charts.
    Optional `on_step` is v3 checkpointing (every N steps).
    """
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()
            if on_step is not None:
                on_step(i, train_state)

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            step_loss = float(loss / batch.ntokens)
            tok_s = float(tokens / elapsed) if elapsed > 0 else 0.0
            print(
                "Step %d | Loss %.2f | Speed %.0f tokens/sec | Learning rate %s"
                % (i, step_loss, tok_s, human_lr(lr))
            )
            if history is not None:
                history.append(
                    {
                        "epoch": epoch,
                        "batch": i,
                        "global_step": train_state.step,
                        "loss": step_loss,
                        "tokens_per_sec": tok_s,
                        "lr": float(lr),
                    }
                )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    mean_loss = total_loss / total_tokens
    return mean_loss, train_state

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )

def example_learning_schedule():
    opts = [
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    # we have 3 examples in opts list.
    for idx, example in enumerate(opts):
        # run 20000 epoch for each example
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )
        tmp = []
        # take 20K dummy training steps, save the learning rate at each step
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    # Enable altair to handle more than 5000 rows
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size_warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model_size_warmup:N")
        .interactive()
    )


show_example(example_learning_schedule)

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())

# Example of label smoothing.


def example_label_smoothing():
    crit = LabelSmoothing(5, 0, 0.4)
    predict = torch.FloatTensor(
        [
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
            [0, 0.2, 0.7, 0.1, 0],
        ]
    )
    crit(x=predict.log(), target=torch.LongTensor([2, 1, 0, 3, 3]))
    LS_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "target distribution": crit.true_dist[x, y].flatten(),
                    "columns": y,
                    "rows": x,
                }
            )
            for y in range(5)
            for x in range(5)
        ]
    )

    return (
        alt.Chart(LS_data)
        .mark_rect(color="Blue", opacity=1)
        .properties(height=200, width=200)
        .encode(
            alt.X("columns:O", title=None),
            alt.Y("rows:O", title=None),
            alt.Color(
                "target distribution:Q", scale=alt.Scale(scheme="viridis")
            ),
        )
        .interactive()
    )


show_example(example_label_smoothing)

def loss(x, crit):
    d = x + 3 * 1
    predict = torch.FloatTensor([[0, x / d, 1 / d, 1 / d, 1 / d]])
    return crit(predict.log(), torch.LongTensor([1])).data


def penalization_visualization():
    crit = LabelSmoothing(5, 0, 0.1)
    loss_data = pd.DataFrame(
        {
            "Loss": [loss(x, crit) for x in range(1, 100)],
            "Steps": list(range(99)),
        }
    ).astype("float")

    return (
        alt.Chart(loss_data)
        .mark_line()
        .properties(width=350)
        .encode(
            x="Steps",
            y="Loss",
        )
        .interactive()
    )


show_example(penalization_visualization)

def data_gen(V, batch_size, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys

# Train the simple copy task.


def example_simple_model():
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    model = make_model(V, V, N=2)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model_size=model.src_embed[0].d_model, factor=1.0, warmup=400
        ),
    )

    batch_size = 80
    for epoch in range(20):
        model.train()
        run_epoch(
            data_gen(V, batch_size, 20),
            model,
            SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train",
        )
        model.eval()
        run_epoch(
            data_gen(V, batch_size, 5),
            model,
            SimpleLossCompute(model.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )[0]

    model.eval()
    src = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(greedy_decode(model, src, src_mask, max_len=max_len, start_symbol=0))


# execute_example(example_simple_model)

## Part 3: DailyDialog training with resume + chat
# Original Annotated Transformer used Multi30k (de->en).
# v3 uses Hugging Face dataset `roskoN/dailydialog` as next-utterance pairs.
# Checkpoints use native torch.save dicts. Chat uses greedy_decode.
# Visualization matches v2 (EDA, training curves, quality, weights).
# Encoder / decoder / attention are unchanged from the paper.

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit


def get_run_config():
    mode = str(RUN_MODE).lower()
    if mode not in ("local", "local_v4", "full", "kaggle_v4"):
        raise ValueError(
            "RUN_MODE must be 'local', 'local_v4', 'full', or 'kaggle_v4', got %r" % mode
        )
    if mode == "local_v4":
        return {
            "mode": "local_v4",
            "train_pairs": None,
            "valid_pairs": None,
            "batch_size": 6,
            "num_epochs": 5,
            "accum_iter": 6,
            "base_lr": 1.0,
            "max_padding": 64,
            "warmup": 3000,
            "N": 4,
            "d_model": 512,
            "d_ff": 2048,
            "h": 8,
            "dropout": 0.1,
            "file_prefix": "v4_local_mixchat_",
            "vocab_path": "v4_vocab_local_mixchat.json",
            "save_every_steps": 2000,
            "train_sources": ["dailydialog", "alpaca"],
            "valid_sources": ["dailydialog"],
            "alpaca_train_pairs": 52000,
            "data_seed": 42,
        }
    if mode == "local":
        return {
            "mode": "local",
            "train_pairs": None,
            "valid_pairs": None,
            "batch_size": 8,
            "num_epochs": 8,
            "accum_iter": 4,
            "base_lr": 1.0,
            "max_padding": 48,
            "warmup": 4000,
            "N": 2,
            "d_model": 512,
            "file_prefix": "dailydialog_local_all8e_",
            "vocab_path": "vocab_local_all.json",
            "save_every_steps": 2500,
            "d_ff": 2048,
            "h": 8,
            "dropout": 0.1,
            "train_sources": ["dailydialog"],
            "valid_sources": ["dailydialog"],
            "alpaca_train_pairs": 0,
            "data_seed": 42,
        }
    if mode == "kaggle_v4":
        # Kaggle 2xT4 profile for stronger encoder-decoder research.
        # Keep micro-batch moderate; use accumulation for stable memory.
        return {
            "mode": "kaggle_v4",
            "train_pairs": None,
            "valid_pairs": None,
            "batch_size": 24,
            "num_epochs": 8,
            "accum_iter": 8,
            "base_lr": 1.0,
            "max_padding": 96,
            "warmup": 10000,
            "N": 6,
            "d_model": 768,
            "d_ff": 3072,
            "h": 12,
            "dropout": 0.1,
            "file_prefix": "v4_kaggle_mixchat_",
            "vocab_path": "v4_vocab_kaggle_mixchat.json",
            "save_every_steps": 1500,
            # Mixed-data strategy: keep DailyDialog for dialogue turns,
            # add Alpaca instruction-response pairs for instruction behavior.
            "train_sources": ["dailydialog", "alpaca"],
            "valid_sources": ["dailydialog"],
            "alpaca_train_pairs": 300000,
            "data_seed": 42,
        }
    return {
        "mode": "full",
        "train_pairs": None,
        "valid_pairs": None,
        "batch_size": 32,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "N": 6,
        "d_model": 512,
        "d_ff": 2048,
        "h": 8,
        "dropout": 0.1,
        "file_prefix": "dailydialog_full_",
        "vocab_path": "vocab.json",
        "save_every_steps": 2000,
        "train_sources": ["dailydialog"],
        "valid_sources": ["dailydialog"],
        "alpaca_train_pairs": 0,
        "data_seed": 42,
    }


def load_tokenizer_en():
    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")
    return spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def normalize_text(text):
    text = re.sub(r"\s+'\s+", "'", text)
    text = re.sub(r"\s+([.,!?])", r"\1", text)
    text = text.replace("。", ".")
    return text.strip()


def build_dailydialog_pairs(split="train", max_pairs=None):
    raw = load_dataset(
        "roskoN/dailydialog",
        revision="refs/convert/parquet",
        token=os.environ.get("HF_TOKEN"),
    )
    pairs = []
    for entry in raw[split]:
        utts = [normalize_text(u) for u in entry["utterances"]]
        for i in range(len(utts) - 1):
            pairs.append((utts[i], utts[i + 1]))
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs
    return pairs


def build_alpaca_pairs(max_pairs=None):
    """Build instruction->response pairs from Alpaca-style data."""
    ds = load_dataset(
        "tatsu-lab/alpaca",
        token=os.environ.get("HF_TOKEN"),
    )["train"]
    pairs = []
    for row in ds:
        inst = normalize_text(str(row.get("instruction", "")).strip())
        extra = normalize_text(str(row.get("input", "")).strip())
        out = normalize_text(str(row.get("output", "")).strip())
        if not out:
            continue
        if extra:
            src = f"{inst}\n{extra}" if inst else extra
        else:
            src = inst
        if src:
            pairs.append((src, out))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def build_train_valid_pairs(config):
    """Build train/valid pairs based on configured source mix."""
    train_sources = list(config.get("train_sources") or ["dailydialog"])
    valid_sources = list(config.get("valid_sources") or ["dailydialog"])
    train_pairs = []
    valid_pairs = []
    train_cap = config.get("train_pairs")
    valid_cap = config.get("valid_pairs")

    if "dailydialog" in train_sources:
        dd_train = build_dailydialog_pairs("train", max_pairs=train_cap)
        train_pairs.extend(dd_train)
    if "alpaca" in train_sources:
        alp_cap = int(config.get("alpaca_train_pairs") or 0)
        if alp_cap > 0:
            train_pairs.extend(build_alpaca_pairs(max_pairs=alp_cap))

    if "dailydialog" in valid_sources:
        dd_valid = build_dailydialog_pairs("validation", max_pairs=valid_cap)
        valid_pairs.extend(dd_valid)

    rng = random.Random(int(config.get("data_seed", 42)))
    rng.shuffle(train_pairs)
    rng.shuffle(valid_pairs)

    if train_cap is not None:
        train_pairs = train_pairs[: int(train_cap)]
    if valid_cap is not None:
        valid_pairs = valid_pairs[: int(valid_cap)]

    print(
        "Data sources -> train: %s | valid: %s"
        % (",".join(train_sources), ",".join(valid_sources)),
        flush=True,
    )
    return train_pairs, valid_pairs


def yield_tokens(pairs, tokenizer):
    for src, tgt in pairs:
        yield tokenize(src, tokenizer)
        yield tokenize(tgt, tokenizer)


def build_vocabulary(spacy_en, pairs):
    print("Building shared English Vocabulary ...")
    tok = Tokenizer(WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = WhitespaceSplit()
    trainer = WordLevelTrainer(
        min_frequency=2,
        special_tokens=["<s>", "</s>", "<blank>", "<unk>"],
    )

    def text_iterator():
        for tokens in yield_tokens(pairs, spacy_en):
            yield " ".join(tokens)

    tok.train_from_iterator(text_iterator(), trainer=trainer)
    return tok


def load_vocab(spacy_en, pairs, vocab_path="vocab.json"):
    if not exists(vocab_path):
        vocab = build_vocabulary(spacy_en, pairs)
        vocab.save(vocab_path)
    else:
        vocab = Tokenizer.from_file(vocab_path)
    print("Finished.\nVocabulary size:")
    print(vocab.get_vocab_size())
    return vocab


def collate_batch(
    batch,
    pipeline,
    vocab,
    device,
    max_padding=128,
):
    bs_id = torch.tensor([vocab.token_to_id("<s>")], device=device)
    eos_id = torch.tensor([vocab.token_to_id("</s>")], device=device)
    pad_id = vocab.token_to_id("<blank>")
    max_body = max(1, max_padding - 2)
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        src_ids = vocab.encode(" ".join(pipeline(_src))).ids[:max_body]
        tgt_ids = vocab.encode(" ".join(pipeline(_tgt))).ids[:max_body]

        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(src_ids, dtype=torch.int64, device=device),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(tgt_ids, dtype=torch.int64, device=device),
                eos_id,
            ],
            0,
        )
        src_list.append(
            pad(
                processed_src,
                (0, max_padding - len(processed_src)),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )
    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)


def create_dataloaders(
    device,
    vocab,
    spacy_en,
    train_pairs,
    valid_pairs,
    batch_size=32,
    max_padding=128,
):
    def pipeline(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            pipeline,
            vocab,
            device,
            max_padding=max_padding,
        )

    train_dataloader = DataLoader(
        train_pairs,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_pairs,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


# ---------------------------------------------------------------------------
# Visualization (charts only — model math stays the 2017 paper Transformer)
# Two data views: (1) repeated conversation text  (2) how sequences look
# After training: loss curves, epoch progress, accuracy vs expected, weights
# ---------------------------------------------------------------------------

VIZ_STATE = {
    "history": [],
    "epoch_rows": [],
    "did_train": False,
}


def display_viz(obj):
    """Show a chart/table in Jupyter; print a text fallback in a script."""
    try:
        from IPython.display import display

        display(obj)
    except Exception:
        if isinstance(obj, pd.DataFrame):
            print(obj.to_string(index=False))
        else:
            print(obj)


def display_note(text):
    try:
        from IPython.display import display, Markdown

        display(Markdown(text))
    except Exception:
        print(text)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def epoch_progress_bar(epoch, num_epochs, width=28):
    done = int(width * (epoch + 1) / max(num_epochs, 1))
    bar = "█" * done + "░" * (width - done)
    return f"[{bar}]  epoch {epoch + 1}/{num_epochs} complete"


def pair_stats_frame(pairs, tokenizer, vocab=None, max_pairs=None):
    """Numeric view of how each utterance looks as a sequence."""
    rows = []
    sample = pairs if max_pairs is None else pairs[:max_pairs]
    for src, tgt in sample:
        src_toks = tokenize(src, tokenizer)
        tgt_toks = tokenize(tgt, tokenizer)
        src_ids = vocab.encode(" ".join(src_toks)).ids if vocab is not None else src_toks
        tgt_ids = vocab.encode(" ".join(tgt_toks)).ids if vocab is not None else tgt_toks
        rows.append(
            {
                "src_chars": len(src),
                "tgt_chars": len(tgt),
                "src_words": len(src_toks),
                "tgt_words": len(tgt_toks),
                "src_tokens": len(src_ids),
                "tgt_tokens": len(tgt_ids),
            }
        )
    return pd.DataFrame(rows)


def explain_pair_stats(stats):
    """Print sentence-length stats in plain English, not raw pandas names."""
    n = int(len(stats))
    src_w = stats["src_words"]
    tgt_w = stats["tgt_words"]
    src_c = stats["src_chars"]
    tgt_c = stats["tgt_chars"]
    lines = [
        "",
        "How long the chat sentences are (%d training pairs)" % n,
        "",
        "                    Previous line          Next reply",
        "Typical (average)   %4.0f words (%3.0f letters)   %4.0f words (%3.0f letters)"
        % (src_w.mean(), src_c.mean(), tgt_w.mean(), tgt_c.mean()),
        "Middle (median)     %4.0f words                   %4.0f words"
        % (src_w.median(), tgt_w.median()),
        "Most are between    %4.0f and %-4.0f words          %4.0f and %-4.0f words"
        % (src_w.quantile(0.25), src_w.quantile(0.75), tgt_w.quantile(0.25), tgt_w.quantile(0.75)),
        "Shortest            %4.0f words                   %4.0f words"
        % (src_w.min(), tgt_w.min()),
        "Longest             %4.0f words                   %4.0f words"
        % (src_w.max(), tgt_w.max()),
        "",
        "Words means tokens after spaCy. Previous line is what the model reads;",
        "next reply is what it learns to write.",
        "",
    ]
    print("\n".join(lines), flush=True)
    display_note(
        "### How long the sentences are\n"
        "Average previous line is **%.0f words**. Average reply is **%.0f words**. "
        "Most turns are between **%.0f and %.0f words**."
        % (src_w.mean(), tgt_w.mean(), src_w.quantile(0.25), src_w.quantile(0.75))
    )


def show_dataset_views(train_pairs, valid_pairs, tokenizer, vocab=None, n_examples=10):
    """Two data views only: repeated chat text, and how those sequences look."""
    alt.data_transformers.disable_max_rows()
    display_note(
        "### Data view 1 — what the conversations look like\n"
        "DailyDialog is everyday chat. Each **row** is one training example: "
        "previous utterance (`src`) → next utterance (`tgt`). "
        "The same short lines (hello, thanks, okay) often **repeat**."
    )
    from collections import Counter

    src_counts = Counter(src for src, _ in train_pairs)
    repeated = [
        {"times": n, "utterance": u}
        for u, n in src_counts.most_common(12)
        if n > 1
    ]
    if not repeated:
        repeated = [{"times": n, "utterance": u} for u, n in src_counts.most_common(8)]
    repeat_df = pd.DataFrame(repeated)
    display_viz(repeat_df)

    examples = []
    for i, (src, tgt) in enumerate(train_pairs[:n_examples]):
        examples.append(
            {
                "pair": i,
                "previous (src)": src,
                "next reply (tgt)": tgt,
            }
        )
    example_df = pd.DataFrame(examples)
    display_note(
        "### Data view 2 — example pairs the model actually trains on\n"
        "This is the seq2seq layout from *Attention Is All You Need*: "
        "encode `src`, decode `tgt`. No extra chat template."
    )
    display_viz(example_df)

    stats = pair_stats_frame(
        train_pairs, tokenizer, vocab=vocab, max_pairs=min(len(train_pairs), 15000)
    )
    display_note(
        "### How those pairs look as sequences\n"
        f"Train pairs: **{len(train_pairs)}** · valid pairs: **{len(valid_pairs)}**. "
        "Bars are word counts after the spaCy tokenizer used by the paper notebook."
    )

    src_hist = (
        alt.Chart(stats)
        .mark_bar(opacity=0.85, color="#2563eb")
        .encode(
            alt.X("src_words:Q", bin=alt.Bin(maxbins=24), title="Words in previous utterance"),
            alt.Y("count()", title="Pairs"),
            tooltip=["count()"],
        )
        .properties(title="Source length", width=360, height=220)
    )
    tgt_hist = (
        alt.Chart(stats)
        .mark_bar(opacity=0.85, color="#db2777")
        .encode(
            alt.X("tgt_words:Q", bin=alt.Bin(maxbins=24), title="Words in next reply"),
            alt.Y("count()", title="Pairs"),
            tooltip=["count()"],
        )
        .properties(title="Target length", width=360, height=220)
    )
    scatter = (
        alt.Chart(stats.sample(n=min(len(stats), 800), random_state=0))
        .mark_circle(size=36, opacity=0.35)
        .encode(
            alt.X("src_words:Q", title="Source words"),
            alt.Y("tgt_words:Q", title="Target words"),
            alt.Color("src_tokens:Q", scale=alt.Scale(scheme="plasma"), title="Src tokens"),
            tooltip=["src_words", "tgt_words", "src_tokens", "tgt_tokens"],
        )
        .properties(title="Source vs target length", width=360, height=220)
    )
    display_viz(alt.hconcat(src_hist, tgt_hist, scatter).resolve_scale(color="independent"))

    corr = stats.corr(numeric_only=True).reset_index().melt("index")
    corr.columns = ["row", "col", "correlation"]
    corr_chart = (
        alt.Chart(corr)
        .mark_rect()
        .encode(
            alt.X("col:O", title=None),
            alt.Y("row:O", title=None),
            alt.Color(
                "correlation:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                title="corr",
            ),
            tooltip=["row", "col", alt.Tooltip("correlation:Q", format=".2f")],
        )
        .properties(title="Length correlations (src/tgt chars, words, tokens)", width=320, height=280)
    )
    display_note(
        "Longer previous turns tend to get slightly longer replies. "
        "That correlation is in the **data**, not added to the model."
    )
    display_viz(corr_chart)
    explain_pair_stats(stats)
    return example_df, stats


def show_training_charts(history, epoch_rows, num_epochs):
    """Charts of loss, learning rate, speed, and how many epochs finished."""
    display_note("### Training progress\nPaper optimizer: Adam + warmup `rate()`. Charts record the same loop.")
    charts = []
    if history:
        hdf = pd.DataFrame(history)
        loss_c = (
            alt.Chart(hdf)
            .mark_line(point=alt.OverlayMarkDef(size=40), strokeWidth=2.2, color="#2563eb")
            .encode(
                alt.X("global_step:Q", title="Optimizer step"),
                alt.Y("loss:Q", title="Batch token loss"),
                tooltip=["epoch", "batch", "global_step", "loss", "lr", "tokens_per_sec"],
            )
            .properties(title="Loss while training", width=400, height=230)
            .interactive()
        )
        lr_c = (
            alt.Chart(hdf)
            .mark_line(strokeWidth=2.2, color="#059669")
            .encode(
                alt.X("global_step:Q", title="Optimizer step"),
                alt.Y("lr:Q", title="Learning rate", axis=alt.Axis(format=".6f")),
                tooltip=["global_step", "lr", "epoch"],
            )
            .properties(title="Learning-rate schedule", width=400, height=230)
            .interactive()
        )
        spd_c = (
            alt.Chart(hdf)
            .mark_area(opacity=0.45, color="#7c3aed")
            .encode(
                alt.X("global_step:Q", title="Optimizer step"),
                alt.Y("tokens_per_sec:Q", title="Tokens / sec"),
                tooltip=["global_step", "tokens_per_sec", "epoch"],
            )
            .properties(title="Throughput", width=400, height=230)
            .interactive()
        )
        charts.append(alt.hconcat(loss_c, lr_c, spd_c))

        if len(hdf) >= 4:
            step_corr = hdf[["loss", "lr", "tokens_per_sec", "global_step"]].corr()
            melt = step_corr.reset_index().melt("index")
            melt.columns = ["row", "col", "correlation"]
            charts.append(
                alt.Chart(melt)
                .mark_rect()
                .encode(
                    alt.X("col:O", title=None),
                    alt.Y("row:O", title=None),
                    alt.Color(
                        "correlation:Q",
                        scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                    ),
                    tooltip=["row", "col", alt.Tooltip("correlation:Q", format=".2f")],
                )
                .properties(title="Training-curve correlations", width=260, height=230)
            )

    if epoch_rows:
        edf = pd.DataFrame(epoch_rows)
        edf["epoch_label"] = edf["epoch"].map(lambda e: f"epoch {int(e)}")
        fold = edf.melt(
            id_vars=["epoch", "epoch_label"],
            value_vars=["train_loss", "val_loss"],
            var_name="split",
            value_name="loss",
        )
        epoch_c = (
            alt.Chart(fold)
            .mark_bar(size=28, cornerRadius=3)
            .encode(
                alt.X("epoch_label:N", title="Completed epoch"),
                alt.Y("loss:Q", title="Mean token loss"),
                alt.Color(
                    "split:N",
                    scale=alt.Scale(
                        domain=["train_loss", "val_loss"],
                        range=["#2563eb", "#f59e0b"],
                    ),
                    legend=alt.Legend(title="Split"),
                ),
                alt.XOffset("split:N"),
                tooltip=["epoch", "split", alt.Tooltip("loss:Q", format=".3f")],
            )
            .properties(
                title=f"Train vs valid loss · {len(edf)}/{num_epochs} epochs done",
                width=max(220, 90 * len(edf)),
                height=230,
            )
        )
        done_n = int(len(edf))
        prog_df = pd.DataFrame(
            [
                {"status": "complete", "epochs": done_n},
                {"status": "remaining", "epochs": max(num_epochs - done_n, 0)},
            ]
        )
        prog_c = (
            alt.Chart(prog_df)
            .mark_bar(size=42, cornerRadius=4)
            .encode(
                alt.Y("status:N", title=None, sort=["complete", "remaining"]),
                alt.X("epochs:Q", title="Epochs"),
                alt.Color(
                    "status:N",
                    scale=alt.Scale(
                        domain=["complete", "remaining"],
                        range=["#059669", "#e5e7eb"],
                    ),
                    legend=None,
                ),
                tooltip=["status", "epochs"],
            )
            .properties(
                title=f"Epoch progress  {done_n}/{num_epochs}",
                width=360,
                height=120,
            )
        )
        charts.append(alt.hconcat(epoch_c, prog_c))

    if not charts:
        print("No training history yet (model was loaded, not trained this run).")
        return
    display_viz(alt.vconcat(*charts))


def weight_table(model):
    """Per-tensor weight stats. Architecture is unchanged; this only reads tensors."""
    rows = []
    for name, p in _unwrap_model(model).named_parameters():
        w = p.detach().float().cpu().reshape(-1)
        rows.append(
            {
                "parameter": name,
                "shape": "x".join(str(s) for s in p.shape),
                "n_weights": int(p.numel()),
                "mean": float(w.mean()),
                "std": float(w.std()) if w.numel() > 1 else 0.0,
                "l2": float(w.norm()),
                "abs_max": float(w.abs().max()),
            }
        )
    return pd.DataFrame(rows)


def show_weight_dashboard(model):
    display_note(
        "### How many weights, and how they look\n"
        "Same Encoder / Decoder / Multi-Head Attention as the paper. "
        "These plots only **inspect** `named_parameters()`."
    )
    wdf = weight_table(model)
    total = int(wdf["n_weights"].sum())
    print(f"Total trainable weights: {total:,}")

    def bucket(name):
        n = name.lower()
        if n.startswith("src_embed") or n.startswith("tgt_embed"):
            return "embeddings + PE"
        if "attn" in n:
            return "attention"
        if "feed_forward" in n or ".w_1" in n or ".w_2" in n:
            return "feed-forward"
        if "norm" in n:
            return "layer-norm"
        if n.startswith("generator"):
            return "generator"
        if n.startswith("encoder"):
            return "encoder (other)"
        if n.startswith("decoder"):
            return "decoder (other)"
        return "other"

    wdf["part"] = wdf["parameter"].map(bucket)
    parts = wdf.groupby("part", as_index=False)["n_weights"].sum()
    part_chart = (
        alt.Chart(parts)
        .mark_bar(cornerRadius=4, color="#4f46e5")
        .encode(
            alt.Y("part:N", sort="-x", title=None),
            alt.X("n_weights:Q", title="Number of weights"),
            tooltip=["part", "n_weights"],
        )
        .properties(title="Weights by block", width=420, height=180)
    )

    top = wdf.sort_values("n_weights", ascending=False).head(18)
    top_chart = (
        alt.Chart(top)
        .mark_bar(cornerRadius=2, color="#0ea5e9")
        .encode(
            alt.Y("parameter:N", sort="-x", title=None),
            alt.X("n_weights:Q", title="Weights"),
            alt.Color("std:Q", scale=alt.Scale(scheme="plasma"), title="std"),
            tooltip=["parameter", "shape", "n_weights", "mean", "std", "l2"],
        )
        .properties(title="Largest parameter tensors", width=480, height=360)
    )
    display_viz(alt.hconcat(part_chart, top_chart))

    # Layer-wise L2 and multi-correlation across encoder/decoder stacks
    layer_rows = []
    core = _unwrap_model(model)
    for stack_name, stack in (("encoder", core.encoder), ("decoder", core.decoder)):
        for i, layer in enumerate(stack.layers):
            flat = torch.cat(
                [p.detach().float().cpu().reshape(-1) for p in layer.parameters()]
            )
            layer_rows.append(
                {
                    "stack": stack_name,
                    "layer": i,
                    "n_weights": int(flat.numel()),
                    "mean": float(flat.mean()),
                    "std": float(flat.std()),
                    "l2": float(flat.norm()),
                }
            )
    ldf = pd.DataFrame(layer_rows)
    layer_chart = (
        alt.Chart(ldf)
        .mark_line(point=True, strokeWidth=2.4)
        .encode(
            alt.X("layer:O", title="Layer index (paper N-stack)"),
            alt.Y("l2:Q", title="L2 of all weights in the layer"),
            alt.Color("stack:N", scale=alt.Scale(range=["#2563eb", "#db2777"])),
            tooltip=["stack", "layer", "n_weights", "mean", "std", "l2"],
        )
        .properties(title="Weight L2 by Transformer layer", width=380, height=230)
        .interactive()
    )

    # Sampled Pearson correlation between layer weight vectors (same length via sample)
    rng = torch.Generator().manual_seed(0)
    samples = {}
    k = 2048
    for stack_name, stack in (("enc", core.encoder), ("dec", core.decoder)):
        for i, layer in enumerate(stack.layers):
            flat = torch.cat(
                [p.detach().float().cpu().reshape(-1) for p in layer.parameters()]
            )
            idx = torch.randint(0, flat.numel(), (k,), generator=rng)
            samples[f"{stack_name}{i}"] = flat[idx]
    names = list(samples.keys())
    mat = torch.stack([samples[n] for n in names], dim=0)
    # correlation across sampled dims
    mat = mat - mat.mean(dim=1, keepdim=True)
    std = mat.std(dim=1, keepdim=True).clamp_min(1e-8)
    corr = (mat / std) @ (mat / std).T / (k - 1)
    corr_rows = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            corr_rows.append({"row": a, "col": b, "correlation": float(corr[i, j])})
    heat = (
        alt.Chart(pd.DataFrame(corr_rows))
        .mark_rect()
        .encode(
            alt.X("col:O", title=None),
            alt.Y("row:O", title=None),
            alt.Color(
                "correlation:Q",
                scale=alt.Scale(scheme="redblue", domain=[-1, 1]),
                title="corr",
            ),
            tooltip=["row", "col", alt.Tooltip("correlation:Q", format=".2f")],
        )
        .properties(
            title="Multi-correlation of sampled layer weights",
            width=280,
            height=280,
        )
    )
    display_viz(alt.hconcat(layer_chart, heat))
    display_viz(wdf.sort_values("n_weights", ascending=False).head(12))
    return wdf


@torch.no_grad()
def token_accuracy(model, dataloader, pad_idx, max_batches=40):
    """Teacher-forced next-token accuracy (argmax vs tgt_y, ignore pad)."""
    model.eval()
    module = _unwrap_model(model)
    correct = 0
    total = 0
    for i, b in enumerate(dataloader):
        batch = Batch(b[0], b[1], pad_idx)
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        pred = module.generator(out).argmax(-1)
        mask = batch.tgt_y != pad_idx
        correct += int((pred[mask] == batch.tgt_y[mask]).sum().item())
        total += int(mask.sum().item())
        if i + 1 >= max_batches:
            break
    acc = correct / max(total, 1)
    return acc, correct, total


def _clean_tokens(tokens, pad_idx, vocab, eos_string="</s>"):
    ids = [t for t in tokens if t != pad_idx]
    text = " ".join(vocab.id_to_token(x) for x in ids)
    return (text.split(eos_string, 1)[0] + " " + eos_string).strip()


def expected_vs_predicted_table(valid_dataloader, model, vocab, n_examples=8):
    """Expected reply vs greedy decode, plus overlap / difference."""
    pad_idx = vocab.token_to_id("<blank>")
    start_symbol = vocab.token_to_id("<s>")
    device = next(model.parameters()).device
    model.eval()
    rows = []
    data_iter = iter(valid_dataloader)
    for idx in range(n_examples):
        b = next(data_iter)
        rb = Batch(b[0], b[1], pad_idx)
        src = rb.src.to(device)
        src_mask = rb.src_mask.to(device)
        expected = _clean_tokens(rb.tgt[0].tolist(), pad_idx, vocab)
        pred_ids = greedy_decode(model, src[:1], src_mask[:1], 64, start_symbol)[0]
        predicted = _clean_tokens(pred_ids.tolist(), pad_idx, vocab)
        exp_set = set(expected.replace("<s>", "").replace("</s>", "").split())
        pred_set = set(predicted.replace("<s>", "").replace("</s>", "").split())
        exp_set.discard("")
        pred_set.discard("")
        inter = exp_set & pred_set
        union = exp_set | pred_set
        overlap = len(inter) / max(len(union), 1)
        missing = " ".join(sorted(exp_set - pred_set))
        extra = " ".join(sorted(pred_set - exp_set))
        rows.append(
            {
                "example": idx,
                "input": _clean_tokens(rb.src[0].tolist(), pad_idx, vocab),
                "expected": expected,
                "predicted": predicted,
                "overlap": round(overlap, 3),
                "missing_from_pred": missing[:120],
                "extra_in_pred": extra[:120],
            }
        )
    return pd.DataFrame(rows)


def show_quality_dashboard(model, vocab, spacy_en, config, valid_pairs, val_loss=None):
    """Accuracy, expected vs predicted, overlap, and what is realistic."""
    display_note(
        "### After training — model quality\n"
        "Teacher-forced **token accuracy** scores next-token argmax vs `tgt_y` "
        "(padding ignored). **Expected** is the true DailyDialog reply. "
        "**Predicted** is paper greedy decode. **Overlap** is Jaccard on words. "
        "**missing / extra** is the word-level difference."
    )
    device = next(model.parameters()).device
    pad_idx = vocab.token_to_id("<blank>")
    vocab_size = vocab.get_vocab_size()
    chance = 1.0 / max(vocab_size, 1)
    _, valid_loader = create_dataloaders(
        device,
        vocab,
        spacy_en,
        valid_pairs,
        valid_pairs,
        batch_size=config["batch_size"],
        max_padding=config["max_padding"],
    )
    acc, correct, total = token_accuracy(model, valid_loader, pad_idx)
    ppl = float(math.exp(min(val_loss, 20))) if val_loss is not None else None
    epochs_done = len(VIZ_STATE.get("epoch_rows") or [])
    if epochs_done == 0 and VIZ_STATE.get("did_train"):
        epochs_done = config["num_epochs"]

    _, single_loader = create_dataloaders(
        device,
        vocab,
        spacy_en,
        valid_pairs,
        valid_pairs,
        batch_size=1,
        max_padding=config["max_padding"],
    )
    cmp = expected_vs_predicted_table(single_loader, model, vocab, n_examples=8)
    mean_overlap = float(cmp["overlap"].mean()) if len(cmp) else 0.0
    exact = int((cmp["overlap"] >= 0.999).sum()) if len(cmp) else 0

    summary = pd.DataFrame(
        [
            {
                "epochs_configured": config["num_epochs"],
                "epochs_logged": epochs_done,
                "N_layers": config["N"],
                "d_model": config["d_model"],
                "vocab_size": vocab_size,
                "valid_pairs": len(valid_pairs),
                "token_accuracy": round(acc, 4),
                "chance_accuracy": round(chance, 6),
                "accuracy_minus_chance": round(acc - chance, 4),
                "correct_tokens": correct,
                "scored_tokens": total,
                "val_loss": None if val_loss is None else round(float(val_loss), 4),
                "perplexity": None if ppl is None else round(ppl, 2),
                "mean_overlap": round(mean_overlap, 3),
                "exact_match_shown": exact,
            }
        ]
    )
    display_viz(summary)

    metric_df = pd.DataFrame(
        [
            {"metric": "token accuracy", "value": acc},
            {"metric": "chance (1 / vocab)", "value": chance},
            {"metric": "mean overlap vs expected", "value": mean_overlap},
        ]
    )
    metric_chart = (
        alt.Chart(metric_df)
        .mark_bar(cornerRadius=4)
        .encode(
            alt.Y("metric:N", sort=["token accuracy", "mean overlap vs expected", "chance (1 / vocab)"], title=None),
            alt.X("value:Q", title="Score", scale=alt.Scale(domain=[0, 1])),
            alt.Color(
                "metric:N",
                scale=alt.Scale(
                    domain=["token accuracy", "mean overlap vs expected", "chance (1 / vocab)"],
                    range=["#2563eb", "#0891b2", "#9ca3af"],
                ),
                legend=None,
            ),
            tooltip=["metric", alt.Tooltip("value:Q", format=".3f")],
        )
        .properties(title="Accuracy vs chance vs overlap", width=480, height=160)
    )

    len_rows = []
    for _, row in cmp.iterrows():
        exp_n = len(str(row["expected"]).split())
        pred_n = len(str(row["predicted"]).split())
        len_rows.append(
            {
                "example": int(row["example"]),
                "expected_words": exp_n,
                "predicted_words": pred_n,
                "word_diff": pred_n - exp_n,
            }
        )
    len_df = pd.DataFrame(len_rows)
    len_fold = len_df.melt(
        id_vars=["example"],
        value_vars=["expected_words", "predicted_words"],
        var_name="side",
        value_name="words",
    )
    len_chart = (
        alt.Chart(len_fold)
        .mark_bar(size=18, cornerRadius=2)
        .encode(
            alt.X("example:O", title="Example"),
            alt.Y("words:Q", title="Words"),
            alt.Color(
                "side:N",
                scale=alt.Scale(
                    domain=["expected_words", "predicted_words"],
                    range=["#059669", "#db2777"],
                ),
                legend=alt.Legend(title="Reply"),
            ),
            alt.XOffset("side:N"),
            tooltip=["example", "side", "words"],
        )
        .properties(title="Expected vs predicted length", width=420, height=200)
    )
    display_viz(alt.hconcat(metric_chart, len_chart))

    if config["mode"] == "local":
        expected_note = (
            "**What to expect (local half-data):** 8 epochs, N=2, ~38k train pairs. "
            "Token accuracy should beat the old 15k/1-epoch run (~26%), but greedy "
            "replies can still look repetitive. That is normal for this 2017 model."
        )
    else:
        expected_note = (
            "**What to expect (full / Kaggle run):** more data and N=6. Token accuracy "
            "and overlap should beat local mode, still far from a modern chatbot."
        )
    display_note(
        expected_note
        + "\n\n**Difference column:** `missing_from_pred` = words in the true reply "
        "the model dropped. `extra_in_pred` = words the model added."
    )
    display_viz(cmp)

    overlap_chart = (
        alt.Chart(cmp)
        .mark_bar(cornerRadius=3, color="#0891b2")
        .encode(
            alt.X("example:O", title="Validation example"),
            alt.Y("overlap:Q", title="Token overlap vs expected", scale=alt.Scale(domain=[0, 1])),
            tooltip=["example", "overlap", "expected", "predicted"],
        )
        .properties(title="Expected vs predicted overlap", width=420, height=220)
    )
    display_viz(overlap_chart)
    print(
        f"Mean overlap on shown examples: {mean_overlap:.3f}  |  "
        f"token accuracy: {acc:.3%}  ({correct}/{total})  |  "
        f"chance: {chance:.3%}  |  lift: {acc - chance:.3%}"
    )
    return summary, cmp


def checkpoint_paths(config):
    prefix = config["file_prefix"]
    return {
        "last": "%slast.pt" % prefix,
        "final": "%sfinal.pt" % prefix,
        "epoch": lambda n: "%sepoch_%d.pt" % (prefix, n),
        "step": lambda n: "%sstep_%d.pt" % (prefix, n),
    }


def collect_rng_state():
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    cpu_state = state["torch"]
    if cpu_state.device.type != "cpu":
        cpu_state = cpu_state.cpu()
    torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and "cuda" in state:
        saved = list(state["cuda"])[: torch.cuda.device_count()]
        cuda_states = [s.cpu().to(torch.uint8) for s in saved]
        torch.cuda.set_rng_state_all(cuda_states)


def build_checkpoint(
    module,
    optimizer,
    scheduler,
    epoch,
    train_state,
    config,
    epoch_done,
    history=None,
    epoch_rows=None,
    val_loss=None,
):
    return {
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": train_state.step,
        "accum_step": train_state.accum_step,
        "samples": train_state.samples,
        "tokens": int(train_state.tokens),
        "epoch_done": epoch_done,
        "config": config,
        "rng": collect_rng_state(),
        "viz_history": list(history or []),
        "epoch_rows": list(epoch_rows or []),
        "val_loss": val_loss,
    }


def save_checkpoint(path, payload):
    torch.save(payload, path)
    print(f"Saved {path}", flush=True)


def cleanup_old_checkpoints(prefix, keep=3):
    """Keep only the latest `keep` step/epoch checkpoints. Always keeps last.pt and final.pt."""
    import glob
    protected = {f"{prefix}last.pt", f"{prefix}final.pt"}
    step_files = sorted(glob.glob(f"{prefix}step_*.pt"), key=os.path.getmtime)
    epoch_files = sorted(glob.glob(f"{prefix}epoch_*.pt"), key=os.path.getmtime)
    for flist in [step_files, epoch_files]:
        while len(flist) > keep:
            old = flist.pop(0)
            if old not in protected:
                os.remove(old)
                print(f"Deleted old checkpoint: {old}", flush=True)


def load_checkpoint_file(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def unwrap_model_state(obj):
    "Accept v3 checkpoint dict or a raw state_dict from v2."
    if isinstance(obj, dict) and "model" in obj and "optimizer" in obj:
        return obj["model"], obj
    return obj, None


def train_worker(vocab, spacy_en, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Train worker using device: {device}", flush=True)
    print(f"RUN_MODE={config['mode']} N={config['N']} epochs={config['num_epochs']}", flush=True)

    pad_idx = vocab.token_to_id("<blank>")
    d_model = config["d_model"]
    vocab_size = vocab.get_vocab_size()
    paths = checkpoint_paths(config)
    save_every = int(config.get("save_every_steps", 200))

    model = make_model(
        vocab_size,
        vocab_size,
        N=config["N"],
        d_model=d_model,
        d_ff=int(config.get("d_ff", 2048)),
        h=int(config.get("h", 8)),
        dropout=float(config.get("dropout", 0.1)),
    )
    model.to(device)
    module = model

    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs", flush=True)
        model = nn.DataParallel(model)
        module = model.module

    criterion = LabelSmoothing(size=vocab_size, padding_idx=pad_idx, smoothing=0.1)
    criterion.to(device)

    if VIZ_STATE.get("train_pairs") is not None and VIZ_STATE.get("valid_pairs") is not None:
        train_pairs = VIZ_STATE["train_pairs"]
        valid_pairs = VIZ_STATE["valid_pairs"]
    else:
        train_pairs, valid_pairs = build_train_valid_pairs(config)
    print(f"Train pairs: {len(train_pairs)} | Valid pairs: {len(valid_pairs)}", flush=True)

    train_dataloader, valid_dataloader = create_dataloaders(
        device,
        vocab,
        spacy_en,
        train_pairs,
        valid_pairs,
        batch_size=config["batch_size"],
        max_padding=config["max_padding"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )

    train_state = TrainState()
    history = []
    epoch_rows = []
    last_val_loss = None
    start_epoch = 0
    resume_path = paths["last"]
    if exists(resume_path):
        print(f"Resuming from {resume_path}", flush=True)
        ckpt = load_checkpoint_file(resume_path, map_location=device)
        model_state, full = unwrap_model_state(ckpt)
        module.load_state_dict(model_state)
        if full is not None:
            optimizer.load_state_dict(full["optimizer"])
            lr_scheduler.load_state_dict(full["scheduler"])
            train_state.step = full.get("step", 0)
            train_state.accum_step = full.get("accum_step", 0)
            train_state.samples = full.get("samples", 0)
            train_state.tokens = full.get("tokens", 0)
            restore_rng_state(full.get("rng"))
            history = list(full.get("viz_history") or [])
            epoch_rows = list(full.get("epoch_rows") or [])
            last_val_loss = full.get("val_loss")
            done = bool(full.get("epoch_done", False))
            start_epoch = full["epoch"] + (1 if done else 0)
            print(
                f"Resume at epoch {start_epoch} (saved epoch={full['epoch']}, "
                f"epoch_done={done}, step={train_state.step})",
                flush=True,
            )
        else:
            print("Loaded raw state_dict; training epochs restart at 0 with these weights.", flush=True)

    current_epoch = start_epoch
    finished = False

    def pack_ckpt(epoch, epoch_done, state=None):
        return build_checkpoint(
            module,
            optimizer,
            lr_scheduler,
            epoch,
            state if state is not None else train_state,
            config,
            epoch_done,
            history=history,
            epoch_rows=epoch_rows,
            val_loss=last_val_loss,
        )

    def write_last(epoch, epoch_done):
        save_checkpoint(paths["last"], pack_ckpt(epoch, epoch_done))

    try:
        for epoch in range(start_epoch, config["num_epochs"]):
            current_epoch = epoch
            model.train()
            print(f"Epoch {epoch} Training ====", flush=True)

            def on_step(batch_i, state):
                if save_every > 0 and state.step > 0 and state.step % save_every == 0:
                    save_checkpoint(
                        paths["step"](state.step),
                        pack_ckpt(epoch, False, state=state),
                    )
                    write_last(epoch, False)
                    cleanup_old_checkpoints(config["file_prefix"], keep=3)

            train_loss, train_state = run_epoch(
                (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
                model,
                SimpleLossCompute(module.generator, criterion),
                optimizer,
                lr_scheduler,
                mode="train+log",
                accum_iter=config["accum_iter"],
                train_state=train_state,
                on_step=on_step,
                history=history,
                epoch=epoch,
            )
            train_loss = float(train_loss)

            try:
                for g in GPUtil.getGPUs():
                    print(
                        "GPU %s: %d%% busy, memory %d MB of %d MB"
                        % (
                            g.name,
                            int(g.load * 100),
                            int(g.memoryUsed),
                            int(g.memoryTotal),
                        ),
                        flush=True,
                    )
            except Exception:
                pass

            print(f"Epoch {epoch} Validation ====", flush=True)
            model.eval()
            val_loss, _ = run_epoch(
                (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
                model,
                SimpleLossCompute(module.generator, criterion),
                DummyOptimizer(),
                DummyScheduler(),
                mode="eval",
            )
            val_loss = float(val_loss)
            last_val_loss = val_loss
            print(
                "Validation check (held-out chat pairs): average loss %.2f  (lower is better)"
                % val_loss,
                flush=True,
            )
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "steps": train_state.step,
                    "tokens": train_state.tokens,
                }
            )
            print(epoch_progress_bar(epoch, config["num_epochs"]), flush=True)
            print(
                "Epoch %d of %d done. Train loss %.2f. Valid loss %.2f. "
                "Steps %d. Tokens seen %d."
                % (
                    epoch + 1,
                    config["num_epochs"],
                    train_loss,
                    val_loss,
                    train_state.step,
                    int(train_state.tokens),
                ),
                flush=True,
            )
            show_training_charts(history, epoch_rows, config["num_epochs"])

            save_checkpoint(paths["epoch"](epoch), pack_ckpt(epoch, True))
            write_last(epoch, True)
            cleanup_old_checkpoints(config["file_prefix"], keep=3)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        save_checkpoint(
            paths["final"],
            pack_ckpt(config["num_epochs"] - 1, True),
        )
        write_last(config["num_epochs"] - 1, True)
        finished = True
        print("=" * 60, flush=True)
        print(
            "Training complete. Saved chat model: %s" % paths["final"],
            flush=True,
        )
        print(
            f"Epochs finished: {config['num_epochs']}/{config['num_epochs']}  "
            f"final val_loss={last_val_loss}",
            flush=True,
        )
        print("=" * 60, flush=True)
    except KeyboardInterrupt:
        print("Interrupted. Writing last checkpoint...", flush=True)
        write_last(current_epoch, False)
        raise
    finally:
        if not finished:
            write_last(current_epoch, False)

    VIZ_STATE["history"] = history
    VIZ_STATE["epoch_rows"] = epoch_rows
    VIZ_STATE["did_train"] = True
    VIZ_STATE["val_loss"] = last_val_loss
    VIZ_STATE["valid_pairs"] = valid_pairs
    VIZ_STATE["train_pairs"] = train_pairs
    return module


def train_model(vocab, spacy_en, config):
    return train_worker(vocab, spacy_en, config)


def load_model_weights(model, path, map_location="cpu"):
    obj = load_checkpoint_file(path, map_location=map_location)
    state, _ = unwrap_model_state(obj)
    model.load_state_dict(state)
    return model


def load_trained_model(vocab, spacy_en, config=None):
    if config is None:
        config = get_run_config()
    paths = checkpoint_paths(config)
    model_path = paths["final"]
    last_path = paths["last"]

    if exists(model_path):
        print(f"Loading finished model {model_path}", flush=True)
        VIZ_STATE["did_train"] = False
    elif exists(last_path):
        print(f"No final checkpoint. Continuing from {last_path}", flush=True)
        train_model(vocab, spacy_en, config)
        model_path = paths["final"] if exists(paths["final"]) else last_path
    else:
        print("No checkpoint found. Starting training from scratch...", flush=True)
        train_model(vocab, spacy_en, config)
        model_path = paths["final"] if exists(paths["final"]) else last_path

    model = make_model(
        vocab.get_vocab_size(),
        vocab.get_vocab_size(),
        N=config["N"],
        d_model=config["d_model"],
        d_ff=int(config.get("d_ff", 2048)),
        h=int(config.get("h", 8)),
        dropout=float(config.get("dropout", 0.1)),
    )
    load_model_weights(model, model_path, map_location="cpu")
    if not VIZ_STATE.get("did_train"):
        try:
            obj = load_checkpoint_file(model_path, map_location="cpu")
            if isinstance(obj, dict):
                VIZ_STATE["history"] = list(obj.get("viz_history") or [])
                VIZ_STATE["epoch_rows"] = list(obj.get("epoch_rows") or [])
                VIZ_STATE["val_loss"] = obj.get("val_loss")
        except Exception:
            pass
    return model


def check_outputs(
    valid_dataloader,
    model,
    vocab,
    n_examples=5,
    eos_string="</s>",
):
    pad_idx = vocab.token_to_id("<blank>")
    start_symbol = vocab.token_to_id("<s>")
    device = next(model.parameters()).device
    model.eval()
    results = [()] * n_examples
    data_iter = iter(valid_dataloader)
    for idx in range(n_examples):
        print("\nExample %d ========\n" % idx)
        b = next(data_iter)
        rb = Batch(b[0], b[1], pad_idx)
        src = rb.src.to(device)
        src_mask = rb.src_mask.to(device)

        src_tokens = [
            vocab.id_to_token(x) for x in rb.src[0].tolist() if x != pad_idx
        ]
        tgt_tokens = [
            vocab.id_to_token(x) for x in rb.tgt[0].tolist() if x != pad_idx
        ]

        print("Input Utterance    : " + " ".join(src_tokens).replace("\n", ""))
        print("Ground Truth Reply : " + " ".join(tgt_tokens).replace("\n", ""))

        model_out = greedy_decode(
            model, src[:1], src_mask[:1], 64, start_symbol
        )[0]
        model_txt = (
            " ".join(
                [vocab.id_to_token(x) for x in model_out.tolist() if x != pad_idx]
            ).split(eos_string, 1)[0]
            + eos_string
        )
        print("Model Reply        : " + model_txt.replace("\n", ""))
        results[idx] = (rb, src_tokens, tgt_tokens, model_out, model_txt)
    return results


def run_model_example(n_examples=5):
    global vocab, spacy_en, model, RUN_CONFIG

    print("Preparing Data ...")
    valid_pairs = VIZ_STATE.get("valid_pairs")
    if valid_pairs is None:
        _, valid_pairs = build_train_valid_pairs(RUN_CONFIG)
    _, valid_dataloader = create_dataloaders(
        torch.device("cpu"),
        vocab,
        spacy_en,
        valid_pairs,
        valid_pairs,
        batch_size=1,
        max_padding=RUN_CONFIG["max_padding"],
    )

    print("Checking Model Outputs:")
    cpu_model = model.cpu()
    example_data = check_outputs(
        valid_dataloader, cpu_model, vocab, n_examples=n_examples
    )
    return cpu_model, example_data


def encode_user_text(text, vocab, spacy_en, device, max_padding):
    pad_id = vocab.token_to_id("<blank>")
    bos_id = vocab.token_to_id("<s>")
    eos_id = vocab.token_to_id("</s>")
    max_body = max(1, max_padding - 2)
    ids = vocab.encode(" ".join(tokenize(normalize_text(text), spacy_en))).ids[:max_body]
    src = torch.tensor([[bos_id] + ids + [eos_id]], dtype=torch.int64, device=device)
    if src.size(1) < max_padding:
        src = pad(src, (0, max_padding - src.size(1)), value=pad_id)
    src_mask = (src != pad_id).unsqueeze(-2)
    return src, src_mask


def decode_ids(ids, vocab, eos_string="</s>"):
    pad_id = vocab.token_to_id("<blank>")
    bos_id = vocab.token_to_id("<s>")
    tokens = []
    for x in ids:
        if x == pad_id or x == bos_id:
            continue
        tok = vocab.id_to_token(int(x))
        if tok == eos_string:
            break
        tokens.append(tok)
    return " ".join(tokens)


def generate_reply(model, vocab, spacy_en, text, max_len=64, max_padding=48):
    device = next(model.parameters()).device
    model.eval()
    src, src_mask = encode_user_text(text, vocab, spacy_en, device, max_padding)
    start_symbol = vocab.token_to_id("<s>")
    with torch.no_grad():
        out = greedy_decode(model, src, src_mask, max_len, start_symbol)[0]
    return decode_ids(out.tolist(), vocab)


def demo_chat(model, vocab, spacy_en, max_padding=48, max_len=64):
    """Fixed prompts so Kaggle Save Version / batch runs do not block on input()."""
    prompts = [
        "hello",
        "how are you",
        "what are you doing today",
        "thanks a lot",
    ]
    print("Demo chat (non-interactive). This is a 2017 seq2seq reply model, not ChatGPT.")
    for user in prompts:
        reply = generate_reply(
            model, vocab, spacy_en, user, max_len=max_len, max_padding=max_padding
        )
        print("You:", user)
        print("Bot:", reply)
        print()


def chat_loop(model, vocab, spacy_en, max_padding=48, max_len=64):
    """One-turn seq2seq chat. Type quit/exit to stop."""
    print("Chat ready. Type a sentence and press Enter. Type 'quit' to stop.")
    print("This is a 2017 encoder-decoder reply model, not ChatGPT.")
    while True:
        try:
            user = input("You: ").strip()
        except EOFError:
            print()
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break
        reply = generate_reply(
            model, vocab, spacy_en, user, max_len=max_len, max_padding=max_padding
        )
        print("Bot:", reply)


def launch_gradio_chat(model, vocab, spacy_en, max_padding=48, max_len=64):
    """Notebook chat UI with 4 example prompts. Uses the same greedy_decode as chat_loop."""
    try:
        import gradio as gr
    except ImportError:
        import subprocess
        import sys

        print("Installing gradio ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio"])
        import gradio as gr

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    chat_model = model.to(device)
    chat_model.eval()
    examples = [
        "hello",
        "how are you",
        "what are you doing today",
        "thanks a lot",
    ]

    def reply_one(message):
        text = (message or "").strip()
        if not text:
            return "Type a short everyday sentence, then Send."
        return generate_reply(
            chat_model,
            vocab,
            spacy_en,
            text,
            max_len=max_len,
            max_padding=max_padding,
        )

    def submit(user_text, history):
        history = list(history or [])
        text = (user_text or "").strip()
        if not text:
            return history, user_text or ""
        bot = reply_one(text)
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": bot})
        return history, ""

    print("Opening Gradio chat. This is one-turn seq2seq, not ChatGPT.", flush=True)
    with gr.Blocks(title="v3 Transformer chat") as demo:
        gr.Markdown(
            "## Chat with your trained Transformer\n"
            "Type in the box and press **Send**. The 4 buttons are example sentences.\n"
            "Each reply is a new greedy decode (no long memory)."
        )
        try:
            chatbot = gr.Chatbot(label="Conversation", height=380, type="messages")
        except TypeError:
            chatbot = gr.Chatbot(label="Conversation", height=380)
        msg = gr.Textbox(
            label="You",
            placeholder="Type here, then Send",
            lines=2,
        )
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")
        gr.Examples(examples, inputs=msg, label="4 example sentences")
        send.click(submit, [msg, chatbot], [chatbot, msg])
        msg.submit(submit, [msg, chatbot], [chatbot, msg])
        clear.click(lambda: ([], ""), outputs=[chatbot, msg])

    return demo.launch(inline=True, quiet=False)


def should_run_interactive_chat():
    if os.environ.get("ENABLE_CHAT", "1").lower() in ("0", "false", "no"):
        return False
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") == "Batch":
        return False
    return True


def print_step(n, total, title, comment):
    """Banner so a script run is readable step-by-step."""
    bar = "=" * 72
    text = f"**Step {n}/{total}: {title}**\n\n{comment}"
    print(flush=True)
    print(bar, flush=True)
    print(f"STEP {n}/{total}: {title}", flush=True)
    print(comment, flush=True)
    print(bar, flush=True)
    display_note(text)


def visualize_layers(model):
    """Inspect encoder/decoder/attention weights. Does not change the paper model."""
    return show_weight_dashboard(model)


def run_pipeline():
    """v3 entry point. Call this last — it runs every stage in order.

    Same visualization pipeline as v2, plus resume checkpoints and chat.

    1 config → 2 data size → 3 vocab → 4 EDA → 5 train/resume/load
    → 6 training curves → 7 quality → 8 layer weights → 9 examples
    → 10 demo chat → 11 live chat (script only)

    Usage:
        python transformer.py
        # or:
        run_pipeline()
    """
    global vocab, spacy_en, model, RUN_CONFIG

    n = 11

    print_step(
        1,
        n,
        "Config",
        "local = laptop subset. full = Colab/Kaggle paper-size (default on Kaggle).\n"
        "v3 extras: last.pt resume + greedy chat. Paper encoder/decoder unchanged.",
    )
    RUN_CONFIG = get_run_config()
    print("Using config:", RUN_CONFIG, flush=True)
    if running_on_kaggle():
        print(
            "Kaggle: GPU + Internet must be on. Checkpoints → /kaggle/working prefix",
            RUN_CONFIG["file_prefix"],
            flush=True,
        )

    print_step(
        2,
        n,
        "Load data — how much",
        "Mixed chat data for v4 research profile.\n"
        "DailyDialog is next-utterance chat; Alpaca adds instruction-response behavior.",
    )
    spacy_en = load_tokenizer_en()
    train_pairs, valid_pairs = build_train_valid_pairs(RUN_CONFIG)
    VIZ_STATE["train_pairs"] = train_pairs
    VIZ_STATE["valid_pairs"] = valid_pairs
    print(
        f"Train pairs: {len(train_pairs)}  |  Valid pairs: {len(valid_pairs)}  |  "
        f"mode={RUN_CONFIG['mode']}",
        flush=True,
    )

    print_step(
        3,
        n,
        "Vocabulary",
        "Word-level vocab from the training pairs (same tokenizer style as the paper notebook).",
    )
    vocab = load_vocab(
        spacy_en, train_pairs, vocab_path=RUN_CONFIG.get("vocab_path", "vocab.json")
    )

    print_step(
        4,
        n,
        "EDA — how the data looks",
        "View 1: which lines repeat. View 2: example src→tgt pairs, length histograms, "
        "and the src/tgt correlation matrix.",
    )
    show_dataset_views(train_pairs, valid_pairs, spacy_en, vocab=vocab)

    print_step(
        5,
        n,
        "Train, resume, or load",
        "Uses native checkpoint dicts: last.pt / epoch_N.pt / step_N.pt / final.pt.\n"
        "If last.pt exists and final.pt does not, training resumes. "
        "During training you will see epoch progress, loss, LR, tokens/sec.",
    )
    model = load_trained_model(vocab, spacy_en, RUN_CONFIG)
    print(
        "Model ready. Checkpoints use prefix:",
        RUN_CONFIG["file_prefix"],
        flush=True,
    )

    print_step(
        6,
        n,
        "Training curves",
        "Loss, learning-rate schedule, throughput, epoch progress, train vs valid loss.",
    )
    show_training_charts(
        VIZ_STATE.get("history") or [],
        VIZ_STATE.get("epoch_rows") or [],
        RUN_CONFIG["num_epochs"],
    )

    print_step(
        7,
        n,
        "Quality — accuracy vs expected",
        "Token accuracy vs chance, expected vs predicted replies, overlap, "
        "missing/extra words, and length comparison.",
    )
    show_quality_dashboard(
        model,
        vocab,
        spacy_en,
        RUN_CONFIG,
        valid_pairs,
        val_loss=VIZ_STATE.get("val_loss"),
    )

    print_step(
        8,
        n,
        "Visualize layers / weights",
        "How many weights in embeddings, attention, feed-forward, generator; "
        "layer L2 and multi-correlation heatmap.",
    )
    visualize_layers(model)

    print_step(
        9,
        n,
        "Example greedy outputs",
        "A few validation inputs with ground-truth reply vs model reply.",
    )
    run_model_example(n_examples=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    print_step(
        10,
        n,
        "Demo chat",
        "Fixed prompts so Kaggle Save Version does not block on input(). "
        "This is 2017 seq2seq, not ChatGPT.",
    )
    demo_chat(
        model,
        vocab,
        spacy_en,
        max_padding=RUN_CONFIG["max_padding"],
    )

    print_step(
        11,
        n,
        "Live chat",
        "In the notebook, run the last Gradio cell. In a script this opens input() chat.",
    )

    def _in_notebook():
        try:
            from IPython import get_ipython
            return get_ipython() is not None
        except Exception:
            return False

    if _in_notebook():
        print(
            "Training/load done. Run the last notebook cell to open Gradio chat:\n"
            "launch_gradio_chat(model, vocab, spacy_en, max_padding=RUN_CONFIG['max_padding'])",
            flush=True,
        )
    elif should_run_interactive_chat():
        chat_loop(
            model,
            vocab,
            spacy_en,
            max_padding=RUN_CONFIG["max_padding"],
        )

    print(flush=True)
    print("=" * 72, flush=True)
    print(
        "PIPELINE COMPLETE. Checkpoints prefix:",
        RUN_CONFIG["file_prefix"],
        flush=True,
    )
    print("=" * 72, flush=True)
    return model


# Run the full pipeline when this file is executed:
#   python transformer.py
if __name__ == "__main__":
    run_pipeline()
