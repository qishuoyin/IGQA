import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

def compute_importance():
    # ========== 0. Configuration ==========
    # You can change this to others, e.g., "EleutherAI/gpt-neox-20b" or your own GPT-NeoX-GQA
    MODEL_NAME = "EleutherAI/pythia-410m-deduped"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model {MODEL_NAME} on {DEVICE}...")

    # ========== 1. Load existing GPT-NeoX weights ==========
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    model.config.use_cache = False  # For better backward stability

    if getattr(model, "gradient_checkpointing", False):
        model.gradient_checkpointing_disable()

    print("num_layers:", model.config.num_hidden_layers)
    print("num_heads:", model.config.num_attention_heads)
    print("num_kv_heads:",
        getattr(model.config, "num_key_value_heads",
                model.config.num_attention_heads))

    # ========== 2. Create a "normal" eval loss (for sanity check) ==========
    #   Use wikitext-2 validation, concatenated + chunked, standard causal LM loss

    print("Loading wikitext-2 validation for eval...")
    raw_val = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    def make_packed(texts, tokenizer, block_size=1024):
        ids = []
        for t in texts:
            ids += tokenizer(t, add_special_tokens=False)["input_ids"] \
                + [tokenizer.eos_token_id]
        full = (len(ids) // block_size) * block_size
        blocks = [ids[i:i+block_size] for i in range(0, full, block_size)]
        return Dataset.from_dict({"input_ids": blocks})

    val_texts = [ex["text"] for ex in raw_val]
    val_packed = make_packed(val_texts, tok, block_size=1024)

    def collate_lm(examples):
        input_ids = [torch.tensor(e["input_ids"], dtype=torch.long)
                    for e in examples]
        input_ids = torch.stack(input_ids, dim=0)  # Already equal length, no padding needed
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    val_loader = DataLoader(
        val_packed,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_lm,
    )

    @torch.no_grad()
    def eval_loss(model, loader, device="cuda", max_batches=50):
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            inp = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            out = model(input_ids=inp, labels=labels)
            loss = out.loss  # batch mean
            total_loss += loss.item() * inp.numel()
            total_tokens += inp.numel()
        return total_loss / total_tokens

    avg_loss = eval_loss(model, val_loader, device=DEVICE, max_batches=50)
    print(f"[Eval] avg loss ≈ {avg_loss:.4f}, ppl ≈ {torch.exp(torch.tensor(avg_loss)).item():.2f}")

    # ========== 3. Prepare a simple "gradient calibration set" ==========
    #   Still use the first small part of wikitext-2 train
    print("Loading wikitext-2 train for calibration...")
    raw_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:40%]")

    def encode_fn(ex):
        ids = tok(
            ex["text"],
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        )["input_ids"]
        if len(ids) == 0:
            ids = [tok.eos_token_id]
        return {"input_ids": ids}

    encoded = raw_train.map(encode_fn, remove_columns=raw_train.column_names)

    def collate_batch(examples):
        input_ids = [torch.tensor(e["input_ids"], dtype=torch.long)
                    for e in examples]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tok.pad_token_id
        )
        attention_mask = (input_ids != tok.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    calib_loader = DataLoader(
        encoded,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_batch,
    )

    # ========== 4. General version: Compute per-layer KV head importance from grad ==========
    def compute_kv_importance_from_grad(W_grad, config):
        """
        W_grad: [out_features, in_features] = query_key_value.weight.grad

        Supports two cases:
        1) Regular MHA: num_key_value_heads == num_attention_heads,
            out_features = 3 * hidden_size
        2) GQA: num_key_value_heads < num_attention_heads,
            out_features = (n_heads + 2 * n_kv_heads) * head_dim
        """
        hidden_size = config.hidden_size
        n_heads = config.num_attention_heads
        n_kv_heads = getattr(config, "num_key_value_heads", n_heads)
        head_dim = hidden_size // n_heads

        out_features, in_features = W_grad.shape
        assert in_features == hidden_size, f"unexpected in_features: {in_features}"

        # Case 1: No GQA
        if n_kv_heads == n_heads:
            assert out_features == 3 * hidden_size, \
                f"expected 3*hidden_size={3*hidden_size}, got {out_features}"
            grad_q = W_grad[0:hidden_size, :]
            grad_k = W_grad[hidden_size:2*hidden_size, :]
            grad_v = W_grad[2*hidden_size:3*hidden_size, :]

            grad_k = grad_k.view(n_heads, head_dim, hidden_size)
            grad_v = grad_v.view(n_heads, head_dim, hidden_size)

            imp = (grad_k.pow(2).sum(dim=(1, 2)).sqrt() +
                grad_v.pow(2).sum(dim=(1, 2)).sqrt())
            return imp  # [n_heads] == [n_kv_heads]

        # Case 2: GQA


        expected_out = (n_heads + 2 * n_kv_heads) * head_dim
        assert out_features == expected_out, \
            f"expected out_features={expected_out}, got {out_features}"

        grad_3d = W_grad.view(n_heads + 2 * n_kv_heads, head_dim, hidden_size)
        grad_q = grad_3d[0:n_heads]
        grad_k = grad_3d[n_heads:n_heads + n_kv_heads]
        grad_v = grad_3d[n_heads + n_kv_heads:]

        imp = (grad_k.pow(2).sum(dim=(1, 2)).sqrt() +
            grad_v.pow(2).sum(dim=(1, 2)).sqrt())
        return imp  # [n_kv_heads]


    def compute_all_layer_importances(
        model,
        dataloader,
        max_batches=10,
        device="cuda",
    ):
        model.zero_grad()
        model.train()  # For backward
        batches_run = 0

        for batch in dataloader:
            if batches_run >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss
            loss.backward()  # Accumulate gradients

            batches_run += 1
            print(f"[Calib] batch {batches_run}, loss={loss.item():.4f}")

        # Return to eval state
        model.eval()

        all_layer_importances = []

        for layer_id, layer in enumerate(model.gpt_neox.layers):
            attn = layer.attention
            W_grad = attn.query_key_value.weight.grad
            if W_grad is None:
                raise RuntimeError(
                    f"Layer {layer_id} has no grad on query_key_value.weight; "
                    "check that backward has been called."
                )
            kv_imp = compute_kv_importance_from_grad(W_grad, model.config)
            all_layer_importances.append(kv_imp.detach().cpu())

        return all_layer_importances

    print("Computing importance scores...")
    importance_scores = compute_all_layer_importances(
        model,
        calib_loader,
        max_batches=128,
        device=DEVICE,
    )

    # Print first few layers to check
    for layer_id, imp in enumerate(importance_scores):
        imp_list = imp.tolist()
        print(f"Layer {layer_id}:")
        print("  KV head importance:", [round(x, 4) for x in imp_list])

    output_file = "kv_head_importance_neox.pt"
    torch.save(
        {"importance_scores": importance_scores},
        output_file,
    )
    print(f"Saved importance scores to {output_file}")

    # ========== 5. Visualization: Importance heatmap for each layer x head ==========
    # Shape: [num_layers, num_kv_heads]
    imp_mat = torch.stack(importance_scores, dim=0)  # [L, H_kv]

    plt.figure(figsize=(8, 6))
    plt.imshow(imp_mat, aspect="auto")
    plt.colorbar(label="KV head importance")
    plt.xlabel("KV head index")
    plt.ylabel("Layer index")
    plt.title(f"KV head importance heatmap ({MODEL_NAME})")
    plt.tight_layout()
    plot_file = "kv_head_importance_heatmap.png"
    plt.savefig(plot_file, dpi=150)
    print(f"Saved heatmap to {plot_file}")
    plt.close()

if __name__ == "__main__":
    compute_importance()

