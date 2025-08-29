
import torch
import torch.nn as nn
from typing import Iterable
import math

import einx
from einops import rearrange, einsum, reduce

class Linear(nn.Module):

    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device))
        # self.bias = nn.Parameter(torch.empty(out_features, device=device)) # Interesting they say bias is not used in LLM

        init_sigma = math.sqrt(2) / math.sqrt(in_features + out_features)
        nn.init.trunc_normal_(self.weight, mean=0, std=init_sigma, a=-3 * init_sigma, b=3 * init_sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"Input tensor must have last dimension of size {self.in_features}, got {x.shape[-1]}")
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.embedding_tensor = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device))
        nn.init.trunc_normal_(self.embedding_tensor, mean=0, std=1, a=-3, b=3)

    def forward(self, tokens_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_tensor[tokens_ids]

class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        # self.weights = nn.Parameter(torch.ones(d_model, device=device))
        self.gain = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # x shape is (batch_size, sequence_length, d_model)
        mean_squared = reduce(x * x, '... d_model -> ... 1', 'mean')
        rms = torch.sqrt(mean_squared + self.eps)

        result = self.gain * x / rms
        return result.to(in_dtype)

class SiLU(nn.Module):
    def __init__(self, device=None, dtype=None):
        super().__init__()

        self.device = device
        self.dtype = dtype

    def forward(self, x):
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff: int = None, device=None, dtype=None):
        super().__init__()

        self.d_model = d_model
        self.device = device
        self.dtype = dtype
#
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            # Round to nearest multiple of 64 for hardware efficiency
            d_ff = ((d_ff + 63) // 64) * 64

        self.d_ff = d_ff

        self.W1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.W2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.W3 = Linear(d_model, d_ff, device=device, dtype=dtype)

        # SiLU activation
        self.silu = SiLU()

    def forward(self, x: torch.Tensor):
        w1x = self.W1(x)
        silu_w1x = self.silu(w1x)
        w3x = self.W3(x)

        return self.W2(silu_w1x * w3x)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Standard RoPE implementation: θ_i = base^(-2i/d)
        # where i goes from 0 to d/2-1
        i_indices = torch.arange(0, self.d_k, 2, dtype=torch.float32, device=device)
        freqs = 1.0 / (self.theta ** (i_indices / self.d_k))

        # Alternative equivalent formulation:
        # freqs = self.theta ** (-i_indices / self.d_k)

        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)

        # Compute angles: position_i * freq_j for all i,j
        angles = torch.outer(positions, freqs)  # Shape: (max_seq_len, d_k/2)

        cos_i_k = torch.cos(angles)
        sin_i_k = torch.sin(angles)

        self.register_buffer('cos_cached', cos_i_k, persistent=False)
        self.register_buffer('sin_cached', sin_i_k, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        cos_values = self.cos_cached[token_positions]
        sin_values = self.sin_cached[token_positions]

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        x_even_rot = x_even * cos_values - x_odd * sin_values
        x_odd_rot = x_even * sin_values + x_odd * cos_values

        rotated = torch.empty_like(x)
        rotated[..., 0::2] = x_even_rot
        rotated[..., 1::2] = x_odd_rot

        return rotated

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:

    x_max = torch.max(x, dim=dim, keepdim=True)[0]

    x_shifted = x - x_max

    x_exp = torch.exp(x_shifted)

    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)

    return x_exp / x_exp_sum

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:

    d_k = query.size(-1)

    scores = einsum(query, key, "... q_len d_k, ... k_len d_k -> ... q_len k_len") / torch.sqrt(torch.tensor(d_k, dtype=query.dtype))

    if mask is not None:

        scores = scores.masked_fill(~mask, float('-inf'))

    attention_probs = softmax(scores, dim=-1)

    output = einsum(attention_probs, value, "... q_len k_len, ... k_len d_v -> ... q_len d_v")

    return output

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope_theta: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len

        self.d_k = self.d_model // num_heads
        self.d_v = self.d_model // num_heads

        # self.W_q = Linear(self.num_heads * self.d_k, d_model)
        self.W_q = Linear(d_model, self.num_heads * self.d_k)
        self.W_k = Linear(self.num_heads * self.d_k, d_model)
        self.W_v = Linear(self.num_heads * self.d_v, d_model)
        # self.W_o = Linear(d_model, self.num_heads * self.d_v)
        self.W_o = Linear(self.num_heads * self.d_v, d_model)

        self.register_buffer("causal_mask", None, persistent=False)

        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_k=self.d_k,
            max_seq_len=max_seq_len
        )

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or create causal mask for the given sequence length."""
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            # Create lower triangular mask (True for positions we can attend to)
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
            self.register_buffer("causal_mask", mask, persistent=False)

        return self.causal_mask[:seq_len, :seq_len]

    def forward(self, x: torch.Tensor, apply_rope=True, token_positions=None):

        B, S, _ = x.shape
        # compute Q, K, V projectsions

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = rearrange(Q, "batch seq (h d) -> batch seq h d", h=self.num_heads)
        K = rearrange(K, "batch seq (h d) -> batch seq h d", h=self.num_heads)
        V = rearrange(V, "batch seq (h d) -> batch seq h d", h=self.num_heads)

        # apply RoPE
        if apply_rope:
            # Use provided positions or default to sequential positions
            if token_positions is None:
                positions = torch.arange(S, device=x.device)
            else:
                positions = token_positions

            Q_reshaped = rearrange(Q, "batch seq h d -> (batch h) seq d")
            K_reshaped = rearrange(K, "batch seq h d -> (batch h) seq d")

            # print(f"Debug: Q reshaped for RoPE: {Q_reshaped.shape}")
            # print(f"Debug: K reshaped for RoPE: {K_reshaped.shape}")

            # Apply RoPE
            Q_rope = self.rope(Q_reshaped, positions)
            K_rope = self.rope(K_reshaped, positions)

            # Reshape back: (batch, seq_len, num_heads, d_k)
            Q = rearrange(Q_rope, "(batch h) seq d -> batch seq h d", h=self.num_heads)
            K = rearrange(K_rope, "(batch h) seq d -> batch seq h d", h=self.num_heads)

        Q = rearrange(Q, "batch seq h d -> batch h seq d")
        K = rearrange(K, "batch seq h d -> batch h seq d")
        V = rearrange(V, "batch seq h d -> batch h seq d")

        mask = self._get_causal_mask(S, x.device)

        attention_output = scaled_dot_product_attention(Q, K, V, mask=mask)

        attention_output = rearrange(attention_output, "batch h seq d -> batch seq (h d)")

        output = self.W_o(attention_output)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 rope_theta: float = 10000.0,
                 max_seq_len: int = 2048,
                 eps: float = 1e-5):

        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff

        self.norm1 = RMSNorm(d_model, eps=eps)
        self.norm2 = RMSNorm(d_model, eps)

        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len
        )

        self.feed_forward = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor, apply_rope: bool = True, token_positions=None):

        # rmsnorm + MHSA
        norm_x = self.norm1(x)
        y1 = x + self.attention(norm_x, apply_rope=apply_rope, token_positions=token_positions)

        # Rmsnorm + FF/SwiGLU
        norm_y1 = self.norm2(y1)
        y2 = y1 + self.feed_forward(norm_y1)
        return y2

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int = None,
        rope_theta: float = 10000.0,
        eps: float = 1e-5
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.vocab_size = vocab_size

        if d_ff is None:
            d_ff = int(8 / 3 * d_model)
            # Round to nearest multiple of 64 for hardware efficiency
            d_ff = ((d_ff + 63) // 64) * 64

        self.d_ff = d_ff

        self.num_layers = num_layers
        self.rope_theta = rope_theta
        self.context_length = context_length

        self.token_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        # self.positional_embedding = Embedding(num_embeddings=context_length, embedding_dim=d_model)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                rope_theta=rope_theta,
                max_seq_len=context_length,
                eps=eps
            )
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(d_model, eps=eps)

        self.lm_head = Linear(d_model, vocab_size)

        self.softmax = nn.Softmax(dim=-1)

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids=None,
            apply_rope: bool = True,
            return_logits: bool = False):

        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        token_embeds = self.token_embedding(input_ids)

        # pos_embeds = self.positional_embedding(position_ids)

        x = token_embeds  # + pos_embeds

        for block in self.transformer_blocks:
            x = block(x, apply_rope=apply_rope, token_positions=position_ids[0] if apply_rope else None)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        if return_logits:
            return logits
        else:
            return torch.softmax(logits, dim=-1)
