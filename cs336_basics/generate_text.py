import argparse
import json
import os
import time
from typing import List, Optional, Dict, Any, Union

import torch
import numpy as np

from transformer_blocks import TransformerLM, softmax
from bpe_main import BPETokenizer

def load_model_and_tokenizer(checkpoint_path: str, config_path: Optional[str] = None,
                             tokenizer_vocab_path: str = None, tokenizer_merges_path: str = None,
                             special_tokens: List[str] = None) -> tuple:
    """
    Load trained model and tokenizer from checkpoint and config files.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to training config (optional, will try to infer)
        tokenizer_vocab_path: Path to tokenizer vocab file
        tokenizer_merges_path: Path to tokenizer merges file
        special_tokens: List of special tokens

    Returns:
        Tuple of (model, tokenizer, config)
    """
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load config
    if config_path is None:
        # Try to find config in same directory as checkpoint
        checkpoint_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(checkpoint_dir, "config.json")

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        # Fallback default config
        print("Warning: No config file found, using default parameters")
        config = {
            'vocab_size': 50257,
            'context_length': 1024,
            'd_model': 768,
            'num_heads': 12,
            'num_layers': 12,
            'rope_theta': 10000.0
        }

    # Initialize model
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        rope_theta=config.get('rope_theta', 10000.0)
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} parameters")

    # Load tokenizer
    if tokenizer_vocab_path and tokenizer_merges_path:
        print("Loading BPE tokenizer...")
        tokenizer = BPETokenizer.from_files(
            vocab_path=tokenizer_vocab_path,
            merges_path=tokenizer_merges_path,
            special_tokens=special_tokens or ["<|endoftext|>"]
        )
        print(f"Tokenizer loaded with vocab size: {len(tokenizer.vocab)}")
    else:
        print("Warning: No tokenizer paths provided, tokenizer will be None")
        tokenizer = None

    return model, tokenizer, config

def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:

    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    scaled_logits = logits / temperature

    probs = softmax(scaled_logits, dim=-1)

    return probs

def top_p_sampling(probs: torch.Tensor, p: float = 0.9) -> torch.Tensor:

    if not (0 < p <= 1):
        raise ValueError("p must be in range (0, 1]")

    if p == 1.0:
        return probs

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff_mask = cumulative_probs > p

    cutoff_mask[..., 0] = False  # Always keep the most probable token
    sorted_probs[cutoff_mask] = 0.0

    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(dim=-1, index=sorted_indices, src=sorted_probs)

    # Renormalize to ensure it's still a valid probability distribution
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

    return filtered_probs

def sample_next_token(logits: torch.Tensor, temperature: float = 1.0,
                      top_p: float = 1.0, deterministic: bool = False) -> torch.Tensor:

    if deterministic:
        return torch.argmax(logits, dim=-1)

    probs = softmax_with_temperature(logits, temperature)

    if top_p < 1.0:
        probs = top_p_sampling(probs, top_p)

    next_token = torch.multinomial(probs, num_samples=1)

    return next_token.squeeze(-1)

@torch.no_grad()
def generate_text(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    prompt: str = "",
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cpu",
    deterministic: bool = False,
    stop_token: str = "<|endoftext|>",
    show_progress: bool = True
) -> str:
    """
    Generate text completion from a prompt using the trained model.

    Args:
        model: Trained transformer model
        tokenizer: BPE tokenizer
        prompt: Input text to complete
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling (lower = more deterministic)
        top_p: Top-p threshold for nucleus sampling
        device: Device to run inference on
        deterministic: If True, use greedy decoding
        stop_token: Token to stop generation
        show_progress: Whether to show generation progress

    Returns:
        Generated text completion
    """
    model.eval()
    model = model.to(device)

    if prompt:
        input_ids = tokenizer.encode(prompt)
        if not input_ids:
            print("Warning: Empty prompt after tokenization")
            input_ids = []
    else:
        input_ids = []

    if input_ids:
        tokens = torch.tensor([input_ids], dtype=torch.long, device=device)
    else:
        # Start with empty sequence - need to check how model reacts to the empty prompt
        tokens = torch.empty((1, 0), dtype=torch.long, device=device)

    print(f"Prompt tokens: {len(input_ids)}")
    print(f"Generating up to {max_new_tokens} new tokens...")
    print(f"Temperature: {temperature}, Top-p: {top_p}")
    print("-" * 50)

    stop_token_id = None
    if stop_token:
        stop_tokens = tokenizer.encode(stop_token)
        if stop_tokens:
            stop_token_id = stop_tokens[0]  # Use first token of stop sequence

    generated_tokens = []
    start_time = time.time()

    for step in range(max_new_tokens):

        current_length = tokens.size(1)

        max_context = model.context_length
        if current_length >= max_context:
            tokens = tokens[:, -(max_context - 1):]

        logits = model(tokens, return_logits=True)

        next_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

        next_token = sample_next_token(
            next_token_logits,
            temperature=temperature,
            top_p=top_p,
            deterministic=deterministic
        )

        tokens = torch.cat([tokens, next_token.unsqueeze(-1)], dim=-1)
        generated_tokens.append(next_token.item())

        if show_progress and (step + 1) % 10 == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = (step + 1) / elapsed
            print(f"Generated {step + 1}/{max_new_tokens} tokens ({tokens_per_sec:.1f} tok/s)")

        if stop_token_id is not None and next_token.item() == stop_token_id:
            print(f"Stop token encountered at step {step + 1}")
            break

    try:
        generated_text = tokenizer.decode(generated_tokens)
    except Exception as e:
        print(f"Warning: Error decoding tokens: {e}")
        generated_text = f"[Decoding error: {generated_tokens}]"

    full_text = prompt + generated_text

    elapsed_time = time.time() - start_time
    tokens_generated = len(generated_tokens)

    print("-" * 50)
    print("Generation complete!")
    print(f"Tokens generated: {tokens_generated}")
    print(f"Time elapsed: {elapsed_time:.2f}s")
    print(f"Speed: {tokens_generated / elapsed_time:.1f} tokens/second")

    return full_text

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate text using trained transformer model")

    # Model loading
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file (optional)")
    parser.add_argument("--vocab", type=str, required=True,
                        help="Path to tokenizer vocab file")
    parser.add_argument("--merges", type=str, required=True,
                        help="Path to tokenizer merges file")

    # Generation parameters
    parser.add_argument("--prompt", type=str, default="",
                        help="Input prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for sampling (lower = more deterministic)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p threshold for nucleus sampling")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use greedy decoding (deterministic)")

    # System parameters
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (cuda, mps, cpu, or auto)")

    # Saw this somewhere - if you want interactive prompting to generate text. Might be useful in the future to vibecode it

    # parser.add_argument("--interactive", action="store_true",
    #                     help="Start interactive generation session")
    parser.add_argument("--special_tokens", nargs="*", default=["<|endoftext|>"],
                        help="Special tokens for tokenizer")

    return parser.parse_args()

def main():
    """Main inference function."""
    args = parse_arguments()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    try:
        model, tokenizer, config = load_model_and_tokenizer(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            tokenizer_vocab_path=args.vocab,
            tokenizer_merges_path=args.merges,
            special_tokens=args.special_tokens
        )

        if tokenizer is None:
            print("Error: Could not load tokenizer")
            return

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return

    # # Interactive mode
    # if args.interactive:
    #     interactive_generation(model, tokenizer, config, device)
    #     return

    if not args.prompt:
        print("No prompt provided. Use --prompt")
        return

    print(f"Input prompt: '{args.prompt}'")

    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
        deterministic=args.deterministic
    )

    print("\n" + "=" * 60)
    print("FINAL GENERATED TEXT:")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)


if __name__ == "__main__":
    main()
