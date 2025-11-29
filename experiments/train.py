#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import math
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
from contextlib import nullcontext
from tqdm.auto import tqdm

# Add the project root to sys.path so we can import from model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.modeling_igqa import GPTconfig, GPTNeoX
from model.utils import (
    assign_kv_groups_from_importance,
    TrainingTracker,
    SpeedEMA,
    set_seed,
    build_scheduler,
    PackedIterable,
    lm_collate,
    pick_amp_dtype
)

# ================== CONFIG (edit here) ==================
args = {
    # Model
    'n_layers': 24,
    'n_head': 16,
    'n_kvhead': 8,
    'd_model': 768,
    'd_ff': 3072,
    'max_seq_len': 2048,
    'dropout': 0.0,
    'tie_weights': False,

    # Train
    'batch_size': 8,
    'grad_accum': 4,
    'lr': 2e-4,
    'weight_decay': 0.1,
    'warmup_ratio': 0.1,
    'max_steps': 40000,            # try 300 for smoke test
    'clip_grad': 1.0,
    'seed': 42,

    # IO
    'output_dir': 'ckpt-colab-300m-imp',
    'save_every': 2000,
    'log_every': 50,
    'eval_every': 500,
    'resume': '',                  # '', 'latest', or path/to/checkpoint.pt

    # Data
    'buffer_chars': 3_000_000,     # increase for more throughput
    'val_docs': 128,
    
    # Method
    'method': 'importance_score', # 'importance_score' or 'GQA'
    'importance_file': 'kv_head_importance_neox.pt'
}

def run_training():
    set_seed(args['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize training tracker
    tracker = TrainingTracker(args['output_dir'])

    # Tokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.model_max_length = args['max_seq_len']
    tok.pad_token = tok.eos_token

    # Import your GPT (RoPE + GQA)
    cfg = GPTconfig(
        vocal_size=len(tok),
        n_layers=args['n_layers'],
        n_head=args['n_head'],
        n_kvhead=args['n_kvhead'],
        d_model=args['d_model'],
        d_ff=args['d_ff'],
        max_seq_len=args['max_seq_len'],
        dropout=args['dropout'],
        tie_weights=args['tie_weights'],
    )

    now_method = args['method']
    kv_indices = []

    if now_method == "importance_score":
        if not os.path.exists(args['importance_file']):
            raise FileNotFoundError(f"Importance file {args['importance_file']} not found. Run scripts/compute_importance.py first.")
            
        imp_obj = torch.load(args['importance_file'])
        importance_scores = imp_obj["importance_scores"]
        print("Loaded importance scores.")

        for layer_id in range(args['n_layers']):
            head_imp_16 = importance_scores[layer_id]          # [16]

            kv_idx, group_sizes, groups = assign_kv_groups_from_importance(
                head_imp_16,
                num_q_heads=args['n_head'],
                num_kv_heads=args['n_kvhead'],
            )

            print(f"[Layer {layer_id}] group_sizes={group_sizes.tolist()}, kv_idx={kv_idx.tolist()}")
            kv_indices.append(torch.as_tensor(kv_idx.tolist(), dtype=torch.long))

    elif now_method == "GQA":
        for layer_id in range(args['n_layers']):
            kv_indices.append(None)
    else:
        raise ValueError(f"Unknown method: {now_method}")


    model = GPTNeoX(cfg, kv_indices).to(device)


    # Data streams
    fw_edu = load_dataset('HuggingFaceFW/fineweb-edu', split='train', streaming=True)
    fw     = load_dataset('HuggingFaceFW/fineweb',     split='train', streaming=True)
    train_stream = interleave_datasets([fw_edu, fw], probabilities=[0.2, 0.8], seed=args['seed'])

    train_packed = PackedIterable(train_stream, tok, block_size=args['max_seq_len'], buffer_chars=args['buffer_chars'])
    collate_fn = lambda batch: lm_collate(batch, pad_id=tok.eos_token_id, block_size=args['max_seq_len'])
    loader = DataLoader(train_packed, batch_size=args['batch_size'], collate_fn=collate_fn, num_workers=2)

    # Optimizer
    decay, no_decay = set(), set()
    for n, p in model.named_parameters():
        if p.ndim >= 2 and "embedding" not in n:  # Linear/Conv weights
            decay.add(n)
        else:  # No decay for bias, norm, embedding
            no_decay.add(n)
    optim_groups = [
        {"params": [p for n, p in model.named_parameters() if n in decay], "weight_decay": args['weight_decay']},
        {"params": [p for n, p in model.named_parameters() if n in no_decay], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args['lr'], betas=(0.9, 0.95), eps=1e-8)
    scheduler = build_scheduler(optimizer, total_steps=args['max_steps'], warmup_ratio=args['warmup_ratio'])

    # AMP autocast
    amp_dtype, use_bf16 = pick_amp_dtype()
    autocast_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype) if device == 'cuda' else nullcontext()

    # Eval set
    val_texts = []
    try:
        val_texts_stream = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        def _take(stream, n):
            c=0
            for ex in stream:
                t = ex.get('text')
                if t:
                    yield t
                    c+=1
                    if c>=n: break
        val_texts = list(_take(val_texts_stream, args['val_docs']))
    except Exception:
        val_texts = []

    # Resume
    os.makedirs(args['output_dir'], exist_ok=True)
    global_step = 0
    if args['resume']:
        ckpt_path = None
        if args['resume'] == 'latest':
            cands = [x for x in os.listdir(args['output_dir']) if x.startswith('checkpoint-') and x.endswith('.pt')]
            if cands:
                cands.sort(key=lambda s: int(s.split('-')[-1].split('.')[0]))
                ckpt_path = os.path.join(args['output_dir'], cands[-1])
        else:
            ckpt_path = args['resume']
        if ckpt_path and os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            global_step = int(state.get('step', 0))
            # Load training history
            if 'tracker' in state:
                tracker = state['tracker']
            else:
                tracker.load()  # Load history from file
            print(f"[resume] loaded {ckpt_path} at step {global_step}")

    # Initialize progress bar
    pbar = tqdm(total=args['max_steps'], initial=global_step, desc="Training",
                unit="step", dynamic_ncols=True, position=0)

    # Training
    model.train()
    tokens_per_step = args['batch_size'] * args['max_seq_len'] * args['grad_accum']
    ema = SpeedEMA(0.1)
    t0 = time.time()
    accum = 0

    accumulated_loss = 0.0

    # PPL Eval function (defined locally in original script)
    @torch.no_grad()
    def eval_ppl(model, tokenizer, texts, block_size=2048, device="cuda", use_bf16=True):
        model.eval()
        ids = []
        for t in texts:
            ids += tokenizer(t, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        full = (len(ids) // block_size) * block_size
        if full == 0:
            return float("nan"), float("nan")
        ids = torch.tensor(ids[:full], dtype=torch.long, device=device).view(-1, block_size)
        total_loss = 0.0
        amp_dtype_local = torch.bfloat16 if use_bf16 else torch.float16

        for i in range(ids.size(0)):
            inp = ids[i].unsqueeze(0)

            # Critical Fix: Create correct shifted labels
            labels = torch.full_like(inp, -100)
            labels[:, :-1] = inp[:, 1:]  # Consistent with training: labels = input_ids shifted right by 1

            with torch.autocast(device_type="cuda", dtype=amp_dtype_local, enabled=(device=="cuda")):
                _, loss = model(inp, labels=labels)

            total_loss += float(loss)

        model.train()
        mean_loss = total_loss / ids.size(0)
        return math.exp(mean_loss), mean_loss


    for batch in loader:
        if global_step >= args['max_steps']:
            break

        input_ids = batch['input_ids'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=(device=='cuda')):
            _, loss = model(input_ids, labels=labels)
            loss = loss / args['grad_accum']  # Normalize loss for gradient accumulation

        loss.backward()
        accumulated_loss += loss.item()  # Accumulate normalized loss
        accum += 1

        if accum % args['grad_accum'] == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip_grad'])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            # Correct loss: Average loss of all micro-batches
            current_loss = accumulated_loss  # Since each loss is already divided by grad_accum, sum is the average

            # Record training loss to tracker
            tracker.add_train_loss(global_step, current_loss)

            # Update progress bar
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'lr': f'{lr:.2e}',
                'step': f'{global_step}/{args["max_steps"]}'
            })

            # Reset accumulator
            accumulated_loss = 0.0

            global_step += 1
            pbar.update(1)

            # Record training loss
            tracker.add_train_loss(global_step, current_loss)

            # logging
            if global_step % args['log_every'] == 0:
                dt = time.time() - t0
                tps = tokens_per_step / max(dt, 1e-6)
                ema_tps = ema.update(tps)

                # Calculate estimated remaining time
                remaining_steps = args['max_steps'] - global_step
                eta_seconds = remaining_steps * (dt / args['log_every'])
                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))

                print(f"\nstep {global_step:6d} | loss {current_loss:.4f} | lr {lr:.2e} | "
                      f"tokens/s {ema_tps:,.0f} | ETA {eta_str} | {(global_step/args['max_steps']):.1%}")
                t0 = time.time()

            # eval
            if args['eval_every'] and (global_step % args['eval_every'] == 0) and val_texts:
                ppl, eval_loss = eval_ppl(model, tok, val_texts, block_size=args['max_seq_len'], device=device, use_bf16=use_bf16)
                tracker.add_eval_loss(global_step, eval_loss, ppl)
                print(f"[eval] step {global_step} | ppl {ppl:.2f} | eval_loss {eval_loss:.4f}")

                # Update loss curve
                tracker.plot_losses()

            # save checkpoint (Improved)
            if args['save_every'] and (global_step % args['save_every'] == 0):
                ckpt_path = os.path.join(args['output_dir'], f"checkpoint-{global_step}.pt")

                # Save complete training state
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'step': global_step,
                    'tracker': tracker,  # Save tracker state
                    'config': args,
                }, ckpt_path)

                # Also save a latest copy
                latest_path = os.path.join(args['output_dir'], "checkpoint-latest.pt")
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'step': global_step,
                    'tracker': tracker,
                    'config': args,
                }, latest_path)

                # Save training history to separate file
                tracker.save()

                print(f"[save] {ckpt_path}")

    pbar.close()

    # final save (Improved)
    final_dir = os.path.join(args['output_dir'], 'final')
    os.makedirs(final_dir, exist_ok=True)

    # Save final model
    torch.save(model.state_dict(), os.path.join(final_dir, 'model.pt'))

    # Save final checkpoint
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': global_step,
        'tracker': tracker,
        'config': args,
    }, os.path.join(final_dir, 'checkpoint-final.pt'))

    # Save tokenizer and configuration
    tok.save_pretrained(final_dir)
    with open(os.path.join(final_dir, 'train_args.json'), 'w') as f:
        json.dump(args, f, indent=2)

    # Generate final loss curve
    tracker.plot_losses()
    tracker.save()

    print(f'Pretrain completed! Final model saved to {final_dir}')

if __name__ == "__main__":
    run_training()

