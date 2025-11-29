import os
import math
import json
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import IterableDataset as TorchIterable
from typing import Iterable, List

# ================== Core Function: Non-uniform KV Expansion ==================

def repeat_kv_nonuniform(k_or_v, kv_idx):
    """
    k_or_v: [B, H_kv, T, D] (Note the dimension order is B, H, T, D)
    kv_idx: [H_q]

    Return:
        expanded_k_or_v: [B, H_q, T, D]
    """
    B, H_kv, T, D = k_or_v.shape
    H_q = kv_idx.numel()

    # Ensure indices are on the correct device
    kv_idx = kv_idx.to(k_or_v.device)

    # Select corresponding KV heads via advanced indexing
    # Result: [B, H_q, T, D]
    expanded_k_or_v = k_or_v[:, kv_idx, :, :]

    return expanded_k_or_v

def assign_kv_groups_from_importance(
    head_imp: torch.Tensor,
    num_q_heads: int = 16,
    num_kv_heads: int = 8,
):
    """
    Given 16 head importance scores for a layer, output:
      - kv_idx: [num_q_heads], which KV group (0..num_kv_heads-1) the q-th Q head uses
      - group_sizes: [num_kv_heads], how many Q heads are in the j-th KV group
    """
    head_imp = torch.as_tensor(head_imp, dtype=torch.float32)
    assert head_imp.numel() == num_q_heads

    # 1) Sort importance descending
    sorted_idx = torch.argsort(head_imp, descending=True)  # [16]

    # 2) Initialize each KV group with one most important head
    groups = [[] for _ in range(num_kv_heads)]
    group_imp_sum = torch.zeros(num_kv_heads, dtype=torch.float32)

    for g in range(num_kv_heads):
        h = sorted_idx[g].item()
        groups[g].append(h)
        group_imp_sum[g] += head_imp[h]

    # 3) Assign remaining heads (lower importance) to the group with current "minimum total importance"
    for idx in sorted_idx[num_kv_heads:]:
        h = idx.item()
        g = torch.argmin(group_imp_sum).item()
        groups[g].append(h)
        group_imp_sum[g] += head_imp[h]

    # 4) Construct kv_idx / group_sizes
    kv_idx = torch.empty(num_q_heads, dtype=torch.long)
    group_sizes = torch.zeros(num_kv_heads, dtype=torch.long)

    for g, heads in enumerate(groups):
        group_sizes[g] = len(heads)
        for h in heads:
            kv_idx[h] = g  # h-th Q head -> uses g-th KV head

    assert kv_idx.numel() == num_q_heads
    assert group_sizes.sum().item() == num_q_heads
    return kv_idx, group_sizes, groups

# ================== Training Utilities ==================

class TrainingTracker:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.train_losses = []
        self.eval_losses = []
        self.eval_ppls = []
        self.steps = []

        # Create plot directory
        self.plot_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)

    def add_train_loss(self, step, loss):
        self.train_losses.append(loss)
        self.steps.append(step)

    def add_eval_loss(self, step, loss, ppl):
        self.eval_losses.append(loss)
        self.eval_ppls.append(ppl)

    def plot_losses(self):
        if len(self.train_losses) < 2:
            return

        plt.figure(figsize=(12, 4))

        # Training Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.steps[:len(self.train_losses)], self.train_losses, 'b-', label='Train Loss')
        if self.eval_losses:
            eval_steps = self.steps[:len(self.eval_losses)]
            plt.plot(eval_steps, self.eval_losses, 'r-', label='Eval Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title('Training and Evaluation Loss')

        # PPL
        plt.subplot(1, 2, 2)
        if self.eval_ppls:
            eval_steps = self.steps[:len(self.eval_ppls)]
            plt.plot(eval_steps, self.eval_ppls, 'g-', label='Eval PPL')
            plt.xlabel('Step')
            plt.ylabel('Perplexity')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.title('Evaluation Perplexity')
            plt.yscale('log')

        plt.tight_layout()
        plot_path = os.path.join(self.plot_dir, f'training_plot_step_{self.steps[-1]}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save(self):
        data = {
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'eval_ppls': self.eval_ppls,
            'steps': self.steps
        }
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        history_path = os.path.join(self.output_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                data = json.load(f)
                self.train_losses = data['train_losses']
                self.eval_losses = data['eval_losses']
                self.eval_ppls = data['eval_ppls']
                self.steps = data['steps']

def set_seed(seed: int):
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SpeedEMA:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.v = None
    def update(self, x):
        self.v = x if self.v is None else (1 - self.alpha) * self.v + self.alpha * x
        return self.v

def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup = int(warmup_ratio * total_steps)
    def lr_lambda(step):
        if warmup > 0 and step < warmup:
            return step / max(1, warmup)
        if total_steps <= warmup:
            return 1.0
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def pick_amp_dtype():
    # Colab: T4/L4 → fp16; A100/H100/L4(bf16-capable) → bf16
    cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0,0)
    # Very rough rule: Ampere(8.x)+ supports bf16 well
    use_bf16 = torch.cuda.is_available() and (cc[0] >= 8)
    return (torch.bfloat16 if use_bf16 else torch.float16), use_bf16

# ================== Data Utilities ==================

def stream_packer(stream: Iterable, tokenizer, block_size=2048, buffer_chars=3_000_000):
    """Concatenate raw texts into a large buffer, tokenize once, cut into fixed blocks."""
    buf = ""
    for ex in stream:
        text = ex.get("text") or ex.get("content") or ""
        if not text:
            continue
        buf += text + "\n"
        if len(buf) >= buffer_chars:
            ids = tokenizer(buf, return_attention_mask=False, add_special_tokens=False)["input_ids"]
            ids += [tokenizer.eos_token_id]
            full = (len(ids) // block_size) * block_size
            for j in range(0, full, block_size):
                yield {"input_ids": ids[j:j+block_size]}
            buf = ""

class PackedIterable(TorchIterable):
    def __init__(self, hf_iterable, tokenizer, block_size=2048, buffer_chars=3_000_000):
        super().__init__()
        self.hf_iterable = hf_iterable
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.buffer_chars = buffer_chars
    def __iter__(self):
        for ex in stream_packer(self.hf_iterable, self.tokenizer, self.block_size, self.buffer_chars):
            yield ex

def lm_collate(batch, pad_id: int, block_size: int):
    ids = [torch.tensor(x["input_ids"], dtype=torch.long) for x in batch]
    x = torch.stack([F.pad(t, (0, block_size - t.size(0)), value=pad_id)[:block_size] for t in ids], dim=0)

    # Target: labels[:, t] = x[:, t+1]; set last position and padding to -100
    labels = torch.full_like(x, -100)
    labels[:, :-1] = x[:, 1:]

    # If next token is pad, do not train that position
    labels[:, :-1][x[:, 1:] == pad_id] = -100

    return {"input_ids": x, "labels": labels}

