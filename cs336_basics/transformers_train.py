import torch
import numpy as np
from typing import Tuple
import torch.nn as nn
import torch.optim as optim
import time

from typing import Union, BinaryIO, IO, Dict, Any, Optional
from pathlib import Path
import os

import argparse
import math
import json
import sys

from transformer_blocks import TransformerLM
from transformer_utils import cross_entropy, AdamW, cos_annealing, gradient_clipping


def load_tokenized_data(file_path: str, dtype: np.dtype = np.uint16) -> np.memmap:
    """Load tokenized data using memory mapping."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Tokenized data file not found: {file_path}")

    try:
        data = np.load(file_path, mmap_mode='r')
        print(f"Loaded memory-mapped data from {file_path}")
        print(f"Shape: {data.shape}, dtype: {data.dtype}")
    except Exception as e:
        print(f"np.load failed ({e}), trying np.memmap...")
        file_size = os.path.getsize(file_path)
        num_tokens = file_size // np.dtype(dtype).itemsize
        data = np.memmap(file_path, dtype=dtype, mode='r', shape=(num_tokens,))
        print(f"Loaded memory-mapped data with shape: {data.shape}, dtype: {data.dtype}")

    return data

def load_batch_old(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:

    if len(x) < context_length + 1:
        raise ValueError(f"Data length {len(x)} is too short for context_length {context_length}")

    max_start_idx = len(x) - context_length

    start_indices = np.random.randint(0, max_start_idx, size=batch_size)

    input_sequences = np.zeros((batch_size, context_length), dtype=x.dtype)
    target_sequences = np.zeros((batch_size, context_length), dtype=x.dtype)

    for i in range(batch_size):
        start_idx = start_indices[i]
        input_sequences[i] = x[start_idx:start_idx + context_length]
        target_sequences[i] = x[start_idx + 1:start_idx + context_length + 1]

    input_tensor = torch.from_numpy(input_sequences).long().to(device)
    target_tensor = torch.from_numpy(target_sequences).long().to(device)

    return input_tensor, target_tensor

def load_batch(x: np.memmap, batch_size: int, context_length: int, device: str):
    """Load a batch of training data."""
    if len(x) < context_length + 1:
        raise ValueError(f"Data length {len(x)} is too short for context_length {context_length}")

    max_start_idx = len(x) - context_length
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)

    input_sequences = []
    target_sequences = []

    for i in range(batch_size):
        start_idx = start_indices[i]
        sequence_slice = x[start_idx:start_idx + context_length + 1]
        input_seq = sequence_slice[:context_length]
        target_seq = sequence_slice[1:context_length + 1]
        input_sequences.append(input_seq)
        target_sequences.append(target_seq)

    input_array = np.stack(input_sequences)
    target_array = np.stack(target_sequences)

    input_tensor = torch.from_numpy(input_array).long().to(device)
    target_tensor = torch.from_numpy(target_array).long().to(device)

    return input_tensor, target_tensor


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]]
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration,
        'timestamp': time.time(),  # Optional: track when checkpoint was saved
    }

    # Save optional metadata if available
    if hasattr(model, 'config'):
        checkpoint['model_config'] = model.config

    try:
        torch.save(checkpoint, out)
        print(f"✓ Checkpoint saved at iteration {iteration}")
        if isinstance(out, (str, os.PathLike)):
            file_size = os.path.getsize(out) / (1024**2)  # Size in MB
            print(f"  File size: {file_size:.1f} MB")
    except Exception as e:
        print(f"✗ Failed to save checkpoint: {e}")
        raise

def load_checkpoint(src, model, optimizer):

    try:
        checkpoint = torch.load(src, map_location='cpu')  # Load to CPU first for device flexibility

        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        iteration = checkpoint['iteration']

        print(f"✓ Checkpoint loaded from iteration {iteration}")
        if 'timestamp' in checkpoint:
            save_time = time.ctime(checkpoint['timestamp'])
            print(f"  Saved at: {save_time}")

        return iteration

    except FileNotFoundError:
        print(f"✗ Checkpoint file not found: {src}")
        raise
    except KeyError as e:
        print(f"✗ Checkpoint missing required key: {e}")
        raise
    except Exception as e:
        print(f"✗ Failed to load checkpoint: {e}")
        raise

def evaluate_model(model: nn.Module, val_data: np.memmap, batch_size: int,
                   context_length: int, device: str, num_eval_batches: int = 10) -> float:
    """Evaluate model on validation data."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(num_eval_batches):
            inputs, targets = load_batch(val_data, batch_size, context_length, device)
            logits = model(inputs, return_logits=True)

            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)

            loss = cross_entropy(logits_flat, targets_flat)
            total_loss += loss.item()

    model.train()
    return total_loss / num_eval_batches

def log_metrics(iteration: int, train_loss: float, val_loss: Optional[float],
                lr: float, elapsed_time: float, tokens_per_sec: float):
    """Log training metrics to console."""
    print(f"Iter {iteration:6d} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Val Loss: {val_loss:.4f} | " if val_loss is not None else ""
          f"LR: {lr:.2e} | "
          f"Time: {elapsed_time:.1f}s | "
          f"Tokens/s: {tokens_per_sec:.0f}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train transformer language model")

    # Data arguments
    parser.add_argument("--train_data", type=str, required=True,
                        help="Path to training data (.npy file)")
    parser.add_argument("--val_data", type=str, required=True,
                        help="Path to validation data (.npy file)")
    parser.add_argument("--vocab_size", type=int, default=50257,
                        help="Vocabulary size")

    # Model arguments
    parser.add_argument("--d_model", type=int, default=768,
                        help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--context_length", type=int, default=1024,
                        help="Maximum context length")
    parser.add_argument("--rope_theta", type=float, default=10000.0,
                        help="RoPE theta parameter")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Maximum training steps")
    parser.add_argument("--max_lr", type=float, default=3e-4,
                        help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=3e-5,
                        help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="Warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.95,
                        help="Adam beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping threshold")

    # Checkpointing and logging
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=1000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_every", type=int, default=500,
                        help="Evaluate every N steps")
    parser.add_argument("--log_every", type=int, default=100,
                        help="Log every N steps")

    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cuda, mps, cpu, or auto)")
    parser.add_argument("--dtype", type=str, default="uint16",
                        help="Data type for tokens (uint16 or uint32)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_arguments()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"CUDA detected: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("Apple Metal Performance Shaders (MPS) detected")
        else:
            device = "cpu"
            print("No GPU acceleration available, using CPU")
    else:
        device = args.device

    print(f"Using device: {device}")

    # Verify device is actually available
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, falling back to CPU")
        device = "cpu"
    elif device == "mps" and not torch.backends.mps.is_available():
        print("Warning: MPS requested but not available, falling back to CPU")
        device = "cpu"

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Save configuration
    config = vars(args)
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {config_path}")

    # Load data
    print("Loading training and validation data...")
    dtype = np.uint16 if args.dtype == "uint16" else np.uint32
    train_data = load_tokenized_data(args.train_data, dtype=dtype)
    val_data = load_tokenized_data(args.val_data, dtype=dtype)

    print(f"Training data: {len(train_data):,} tokens")
    print(f"Validation data: {len(val_data):,} tokens")

    # Initialize model
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        rope_theta=args.rope_theta
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,  # Will be overridden by scheduler
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay
    )

    # Resume from checkpoint if provided
    start_iteration = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from checkpoint: {args.resume_from}")
        start_iteration = load_checkpoint(args.resume_from, model, optimizer)

    # Training loop
    print(f"Starting training from iteration {start_iteration}...")
    print("=" * 80)

    model.train()
    start_time = time.time()
    log_losses = []

    for iteration in range(start_iteration, args.max_steps):
        iter_start_time = time.time()

        # Update learning rate
        lr = cos_annealing(
            t=iteration,
            alpha_max=config['max_lr'],
            alpha_min=config['min_lr'],
            Tw=config['warmup_steps'],
            Tc=config['max_steps']
        )

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Load batch
        inputs, targets = load_batch(train_data, args.batch_size, args.context_length, device)

        # Forward pass
        logits = model(inputs, return_logits=True)

        # Compute loss
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        loss = cross_entropy(logits_flat, targets_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        gradient_clipping(model.parameters(), args.grad_clip)

        # Optimizer step
        optimizer.step()

        # Logging
        iter_time = time.time() - iter_start_time
        tokens_processed = args.batch_size * args.context_length
        tokens_per_sec = tokens_processed / iter_time

        log_losses.append(loss.item())

        if iteration % args.log_every == 0:
            avg_loss = sum(log_losses) / len(log_losses)
            elapsed_time = time.time() - start_time

            val_loss = None
            if iteration % args.eval_every == 0 and iteration > 0:
                print("Evaluating on validation set...")
                val_loss = evaluate_model(model, val_data, args.batch_size,
                                          args.context_length, device)

            log_metrics(iteration, avg_loss, val_loss, lr, elapsed_time, tokens_per_sec)
            log_losses = []

        # Save checkpoint
        if iteration % args.save_every == 0 and iteration > 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_{iteration:06d}.pt")
            save_checkpoint(model, optimizer, iteration, checkpoint_path)

            # Also save as latest checkpoint
            latest_path = os.path.join(args.checkpoint_dir, "latest.pt")
            save_checkpoint(model, optimizer, iteration, latest_path)

    # Final checkpoint
    final_path = os.path.join(args.checkpoint_dir, "final.pt")
    save_checkpoint(model, optimizer, args.max_steps, final_path)

    # Final evaluation
    print("\nFinal evaluation:")
    final_val_loss = evaluate_model(model, val_data, args.batch_size,
                                    args.context_length, device, num_eval_batches=50)
    print(f"Final validation loss: {final_val_loss:.4f}")

    total_time = time.time() - start_time
    total_tokens = args.max_steps * args.batch_size * args.context_length

    print("=" * 80)
    print("Training completed!")
    print(f"Total time: {total_time / 3600:.2f} hours")
    print(f"Total tokens processed: {total_tokens:,}")
    print(f"Average tokens/second: {total_tokens / total_time:.0f}")
    print(f"Checkpoints saved in: {args.checkpoint_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
