# Kaggle Setup Guide — v4 Transformer

## Tokens

### Kaggle API Token (`KAGGLE_API_TOKEN`)

- **Value:** stored in `.env` file
- **Format:** `KGAT_...` (Kaggle API Token format)
- **Requires:** kaggle CLI >= 1.8.0 or kagglehub >= 0.4.1
- **Used for:** Pushing notebooks to Kaggle, managing kernels, downloading competition data

**Setup (one-time):**

```bash
# Option 1: Environment variable
export KAGGLE_API_TOKEN=YOUR_KAGGLE_TOKEN

# Option 2: Save to access_token file (recommended)
mkdir -p ~/.kaggle
echo "YOUR_KAGGLE_TOKEN" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

**Verify it works:**

```bash
kaggle competitions list
```

### Hugging Face Token (`HF_TOKEN`)

- **Value:** stored in `.env` file
- **Format:** `hf_...`
- **Used for:** Downloading datasets from Hugging Face Hub (DailyDialog, Alpaca)
- **Without it:** Downloads are anonymous with lower rate limits and some datasets may fail

**Setup:**

The notebook (cell 2) automatically sets `HF_TOKEN` and calls `huggingface_hub.login()`. No manual setup needed if you run the notebook from top to bottom.

For CLI usage:

```bash
export HF_TOKEN=hf_YOUR_HUGGINGFACE_TOKEN
```

## Pushing the Notebook to Kaggle

### 1. Install kaggle CLI

```bash
pip install kaggle --upgrade
```

Ensure version >= 1.8.0 for API token support:

```bash
kaggle --version
```

### 2. Create `kernel-metadata.json`

This file must be in the same folder as `transformer.ipynb`:

```json
{
  "id": "YOUR_KAGGLE_USERNAME/v4-transformer-chatbot",
  "title": "v4 Transformer Chatbot",
  "code_file": "transformer.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "competition_sources": [],
  "dataset_sources": [],
  "kernel_sources": []
}
```

Replace `YOUR_KAGGLE_USERNAME` with your actual Kaggle username.

### 3. Push

```bash
cd /home/creator/Desktop/Deeplearning/finetuning/version/v4
kaggle kernels push -p .
```

### 4. Monitor

```bash
kaggle kernels status YOUR_KAGGLE_USERNAME/v4-transformer-chatbot
kaggle kernels output YOUR_KAGGLE_USERNAME/v4-transformer-chatbot -p ./output
```

## Kaggle Environment Details

| Setting | Value |
|---------|-------|
| GPU | 2x Tesla T4 (16 GB each) |
| RAM | 16 GB |
| Disk | ~70 GB |
| Runtime limit | 12 hours |
| Internet | Must be enabled (for dataset downloads) |
| GPU quota | 30 hours/week |

## v4 Model Config on Kaggle

| Parameter | Value |
|-----------|-------|
| N (layers) | 6 encoder + 6 decoder |
| d_model | 768 |
| d_ff | 3072 |
| heads | 12 |
| context window | 96 tokens |
| batch_size | 24 |
| accum_iter | 8 (effective batch = 192) |
| warmup | 10,000 steps |
| epochs | 8 |
| data | ~76k DailyDialog + ~52k Alpaca = ~128k pairs |
| parameters | ~130M |

## Files in This Folder

| File | Purpose |
|------|---------|
| `transformer.ipynb` | Main notebook — run on Kaggle |
| `transformer.py` | Python script mirror (same code) |
| `.env` | Stores HF_TOKEN and KAGGLE_API_TOKEN |
| `kaggle.md` | This guide |
| `README.md` | Architecture and research documentation |
| `kernel-metadata.json` | Kaggle push config (create before pushing) |
