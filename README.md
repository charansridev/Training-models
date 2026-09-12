# Training-models

Fine-tuning experiments that turn a small open-weight LLM into a task-specific classifier.

The repo currently holds one end-to-end project: a **financial fraud risk classifier** built by
QLoRA fine-tuning [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) on
transaction records reframed as chat conversations.

| | |
|---|---|
| **Notebook** | [`Financial_Fraud_Detection_System (3).ipynb`](Financial_Fraud_Detection_System%20(3).ipynb) |
| **Base model** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Dataset** | [`CiferAI/Cifer-Fraud-Detection-Dataset-AF`](https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF) (21M transactions) |
| **Method** | 4-bit NF4 quantization + LoRA adapters, supervised fine-tuning via TRL `SFTTrainer` |
| **Trained adapter** | [`devonfire/Financial-Fraud-Detection-Model-Qwen-1.5b`](https://huggingface.co/devonfire/Financial-Fraud-Detection-Model-Qwen-1.5b) |
| **Hardware** | Single Colab T4 (16 GB), ~31 min for 3 epochs |

---

## What the model does

It takes a plain-text description of a money transfer and replies with a single risk label —
`HIGH` or `LOW`.

**Input**

```
Analyze this transaction for fraud risk:
- Type: TRANSFER
- Amount: $85,000.00
- Sender Balance Before: $85,000.00
- Sender Balance After: $0.00
- Recipient Balance Before: $0.00
- Recipient Balance After: $0.00
```

**Output**

```
LOW
```

Framing tabular fraud detection as a chat task is the whole point of the experiment: instead of
training a gradient-boosted tree on six numeric columns, each row is rendered into natural language
and the model learns the label as the assistant's next turn.

---

## Pipeline

The notebook runs top to bottom in eight stages.

**1 · Load and subsample.** The source dataset has 21,000,000 transactions; the notebook takes the
first 1,000,000 to keep download and filtering time reasonable.

**2 · Handle the class imbalance.** Fraud is rare — in that slice, 1,313 fraudulent vs. 998,687
legitimate transactions (~0.13%). Training on that distribution would teach the model to always
answer `LOW`, so the notebook samples 500 of each and shuffles them into a balanced 1,000-row set.

**3 · Convert to conversations.** Each row becomes a `{"messages": [user, assistant]}` pair. The
user turn describes the transaction type, amount, and the four before/after balance fields; the
assistant turn is `HIGH` when `isFraud == 1`, otherwise `LOW`.

**4 · Quantize the base model.** `BitsAndBytesConfig` loads Qwen2.5-1.5B in 4-bit NF4 with
`bfloat16` compute, which is what lets a 1.5B model train inside a free T4.

**5 · Attach LoRA adapters.**

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

Only the attention projections are adapted — the base weights stay frozen, so the trained artifact
is an 8.75 MB adapter rather than a 3 GB checkpoint.

**6 · Train.** `SFTTrainer` with 3 epochs, batch size 4, gradient accumulation 2, LR 2e-4,
`max_length` 512 → 375 optimizer steps.

**7 · Push to the Hub.** The final checkpoint's adapter, tokenizer, and chat template are uploaded;
optimizer/scheduler/RNG state are excluded via `ignore_patterns`.

**8 · Reload and test.** A `text-generation` pipeline pulls the adapter back from the Hub and runs
two sample transactions through it.

---

## Training results

| Metric | Value |
|---|---|
| Steps | 375 (3 epochs) |
| Final training loss | 0.7840 |
| Mean token accuracy | 0.7230 |
| Entropy | 0.7348 |
| Runtime | 1,847 s (~31 min) |
| Tokens seen | 389,910 |

Loss drops steeply early (2.04 → 0.85 by step 30) and then flattens around 0.73–0.74 for the rest
of training — the model learns the output *format* almost immediately and makes little measurable
progress on the actual classification after that.

---

## Running it

The notebook was written for Google Colab with a T4 GPU and installs its own dependencies:

```bash
pip install -qU datasets transformers bitsandbytes accelerate peft trl
```

Open the notebook in Colab (Runtime → Change runtime type → **T4 GPU**) and run the cells in order.
Two things to be aware of before the Hub cells:

- **`repo_name` is never assigned in the notebook.** The upload/test cells reference it, so define
  it yourself first, e.g. `repo_name = "your-username/Financial-Fraud-Detection-Model-Qwen-1.5b"`.
- **The first Hub cell calls `delete_repo` before `create_repo`.** That is fine for re-running your
  own experiment, but it will destroy an existing repo of that name — drop the `delete_repo` line
  unless you mean it.
- The checkpoint path in `upload_folder` is hardcoded to `/content/fraud-detector/checkpoint-375`.
  It changes if you alter the epoch count or batch size.

### Using the trained adapter

```python
from transformers import pipeline

repo_name = "devonfire/Financial-Fraud-Detection-Model-Qwen-1.5b"
generator = pipeline("text-generation", model=repo_name, device="cuda")

question = """Analyze this transaction for fraud risk:
- Type: TRANSFER
- Amount: $85,000.00
- Sender Balance Before: $85,000.00
- Sender Balance After: $0.00
- Recipient Balance Before: $0.00
- Recipient Balance After: $0.00"""

out = generator([{"role": "user", "content": question}],
                max_new_tokens=150, return_full_text=False)[0]
print(out["generated_text"])   # -> LOW
```

---

## Known limitations

This is a learning project, not a production fraud system. Be honest about what the run shows:

- **The two sample predictions in the notebook look wrong.** A classic account-drain pattern
  (`TRANSFER` of $85,000 emptying the sender to $0 while the recipient balance never moves) is
  labelled `LOW`, while a mundane $150 `PAYMENT` with consistent balances is labelled `HIGH`. Both
  are the opposite of what a working detector should say.
- **No evaluation split.** Training loss is the only signal recorded — there is no held-out
  accuracy, precision/recall, or confusion matrix, so real performance is unmeasured.
- **1,000 training rows out of 21M.** The balanced sample is tiny, and 500 fraud examples is not
  much coverage of fraud behaviour.
- **50/50 balancing distorts the base rate.** It stops the model collapsing to a single label, but
  the resulting classifier is badly calibrated against real-world fraud frequency (~0.13%).
- **The prompt drops available signal.** `step`, `nameOrig`, and `nameDest` are not included, so the
  model can't use timing or counterparty history.

Natural next steps: carve out a test split and report precision/recall on an imbalanced holdout,
train on more data, and compare against a gradient-boosting baseline — if a 1.5B LLM can't beat
LightGBM on six numeric columns, that itself is the interesting result.

---

## Repository layout

```
.
├── Financial_Fraud_Detection_System (3).ipynb   # full pipeline: data → LoRA training → Hub → inference
└── README.md
```
