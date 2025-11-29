from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import repeat_kv_nonuniform

@dataclass
class GPTconfig:
  vocal_size: int = 50257
  n_layers: int = 16
  d_model: int = 1024
  d_ff: int = 4096
  n_head: int = 16
  max_seq_len: int = 2048
  dropout: float = 0.0
  attn_type: str = "full"
  tie_weights: bool = False
  norm_eps: float = 1e-5
  rope_base: int = 10000
  n_kvhead: int = 8
  activation_fn: str = "gelu"

def count_parameters(model: nn.Module):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TokenEmbedding(nn.Module):
  def __init__(self, config: GPTconfig):
    super().__init__()
    self.token_embedding = nn.Embedding(config.vocal_size, config.d_model) ## Converts each token_id in (B, T) to the corresponding D-dimensional embedding, so the output is (B, T, D)
    nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

  def forward(self, x):  #(B,T)
    return self.token_embedding(x) #(B, T, D)

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # Gate branch
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)  # Value branch
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)  # Output projection

        # Add initialization
        self._init_weights()

    def _init_weights(self):
        # Use GPT/LLaMA style initialization
        nn.init.normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w2.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w3.weight, mean=0.0, std=0.02)

    def forward(self, x):
        gate = F.silu(self.w1(x))  # Correction: Use SiLU instead of sigmoid
        value = self.w2(x)
        return self.w3(gate * value)

class RMSNorm(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.eps = config.norm_eps
    self.gamma = nn.Parameter(torch.ones(config.d_model))

  # (B, T, C)
  def forward(self, x):
    rms = (x.pow(2).mean(dim=-1, keepdim=True)+self.eps).sqrt() #(B, T, 1)
    return self.gamma * x / rms

class MLP(nn.Module):
  def __init__(self, config):
    super().__init__()
    if config.activation_fn == "gelu":
        self.fc1 = nn.Linear(config.d_model, config.d_ff)
        self.fc2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        self.activation = "gelu"

    elif config.activation_fn == "swish":
        self.ffn = SwiGLU(config.d_model, int(config.d_ff*2/3))
        self.dropout = nn.Dropout(config.dropout)
        self.activation = "swish"

  def forward(self, x):
    if self.activation == "gelu":
      x = self.fc1(x)
      x = F.gelu(x)
      x = self.fc2(x)
    elif self.activation == "swish":
      x = self.ffn(x)
    return self.dropout(x)


class RotaryEmbedding(nn.Module):
  def __init__(self,config: GPTconfig):
    super().__init__()
    self.config = config
    head_dim = config.d_model // config.n_head
    assert head_dim % 2 == 0, "head_dim must be divisible by 2"
    inv_freq = 1.0 / (config.rope_base ** (torch.arange(0, head_dim, 2)/head_dim))
    pos = torch.arange(config.max_seq_len, dtype=torch.float32)
    freqs = pos[:, None] * inv_freq[None, :]
    self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)
    self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)

  def forward(self, T, device):
    return self.sin_cached[:T, :].to(device), self.cos_cached[:T, :].to(device)

def apply_RoPE(x, cos, sin):
  # x　(B, H, T, D)
  x1 = x[..., ::2] ## Elements 0, 2, 4, ... D-2 in the D dimension (B, H, T, D/2)
  x2 = x[...,1::2] ## Elements 1, 3, 5, ... D-1 in the D dimension (B, H, T, D/2)
  cos = cos[None, None, :, :] #(1, 1, T, D/2)
  sin = sin[None, None, :, :] #(1, 1, T, D/2)
  y1 = x1*cos-x2*sin
  y2 = x1*sin+x2*cos
  y = torch.stack([y1, y2], dim=-1).flatten(-2)
  return y



def _split_heads(x, n_head):
  B, T, D = x.shape
  x = x.view(B, T, n_head, D//n_head).transpose(1,2)
  return x

def _merge_heads(x):
  B, H, T, D =x.shape
  x = x.transpose(1,2).contiguous().view(B, T, D*H)
  return x

class CausalMHA_RoPE_GQA(nn.Module):
  def __init__(self,config: GPTconfig, kv_idx: torch.Tensor = None):
    super().__init__()
    assert config.d_model % config.n_head == 0
    self.n_head = config.n_head
    self.n_kvheads = config.n_kvhead if config.n_kvhead else config.n_head
    assert self.n_head % self.n_kvheads == 0, "n_head must be divisible by n_kvheads"
    self.head_dim = config.d_model // config.n_head
    self.kv_proj_dim = self.head_dim * self.n_kvheads
    self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
    self.k_proj = nn.Linear(config.d_model, self.kv_proj_dim, bias=False)
    self.v_proj = nn.Linear(config.d_model, self.kv_proj_dim, bias=False)
    self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
    self.attn_dropout = nn.Dropout(config.dropout)
    self.rope_emb = RotaryEmbedding(config)

    for mod in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
      nn.init.normal_(mod.weight, mean=0.0, std=0.02)


    # <-- 2. Register kv_idx
    if kv_idx is not None:
        assert kv_idx.numel() == config.n_head
        self.register_buffer("kv_idx", kv_idx, persistent=False)
    else:
        # If not provided, kv_idx remains None, use standard uniform GQA
        self.kv_idx = None


  def forward(self, x):
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)
    q = _split_heads(q, self.n_head)
    k = _split_heads(k, self.n_kvheads)
    v = _split_heads(v, self.n_kvheads)

    q = apply_RoPE(q, *self.rope_emb(q.shape[2], q.device))
    k = apply_RoPE(k, *self.rope_emb(k.shape[2], k.device))

    if self.n_kvheads != self.n_head:
        if self.kv_idx is not None:
            # 🚀 Use non-uniform GQA
            k = repeat_kv_nonuniform(k, self.kv_idx)
            v = repeat_kv_nonuniform(v, self.kv_idx)
        else:
            # 🤖 Use standard uniform GQA (as backup)
            repeat_times = self.n_head // self.n_kvheads
            k = torch.repeat_interleave(k, repeat_times, dim=1)
            v = torch.repeat_interleave(v, repeat_times, dim=1)

    y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p, is_causal=True)
    y = _merge_heads(y)
    y = self.attn_dropout(self.o_proj(y))
    return y


class TransformerBlock(nn.Module):
  def __init__(self, config: GPTconfig, kv_idx: torch.Tensor = None):
    super().__init__()
    self.attn = CausalMHA_RoPE_GQA(config, kv_idx)
    self.norm1 = RMSNorm(config)
    self.ffn = MLP(config)
    self.norm2 = RMSNorm(config)

  def forward(self, x):
    x = self.attn(self.norm1(x)) + x
    x = self.ffn(self.norm2(x)) + x
    return x


class GPTNeoX(nn.Module):
  def __init__(self, config: GPTconfig, all_kv_indices: list[torch.Tensor] = None):
    super().__init__()
    self.config = config
    self.token_embedder = TokenEmbedding(config)
    self.transformers = nn.ModuleList()
    for i in range(config.n_layers):
        kv_idx = all_kv_indices[i] if all_kv_indices and i < len(all_kv_indices) else None
        self.transformers.append(TransformerBlock(config, kv_idx))
    self.norm = RMSNorm(config)
    self.proj_to_vocab = nn.Linear(config.d_model, config.vocal_size, bias=False)
    nn.init.normal_(self.proj_to_vocab.weight, mean=0.0, std=0.02)

    if config.tie_weights:
      self.proj_to_vocab.weight = self.token_embedder.token_embedding.weight

  def forward(self, input_ids, labels=None):
    B, T = input_ids.shape
    x = self.token_embedder(input_ids)
    for transformer in self.transformers:
      x = transformer(x)
    x = self.norm(x)
    logits = self.proj_to_vocab(x)

    loss = None
    if labels is not None:
      loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

    return logits, loss

