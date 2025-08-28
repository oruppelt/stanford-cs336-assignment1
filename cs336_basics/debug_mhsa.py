
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
        # if d_ff is None:
        d_ff = int(8 / 3 * d_model)
        # Round to nearest multiple of 64 for hardware efficiency
        d_ff = ((d_ff + 63) // 64) * 64

        self.d_ff = d_ff

        self.W1 = nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)
        self.W2 = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.W3 = nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)

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

import torch
import torch.nn as nn
from einops import rearrange, einsum

# Your clean MultiHeadSelfAttention class (without debug prints)
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope_theta: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len

        self.d_k = self.d_model // num_heads
        self.d_v = self.d_model // num_heads

        # Linear projections - using your original Linear class
        self.W_q = Linear(d_model, self.num_heads * self.d_k)
        self.W_k = Linear(d_model, self.num_heads * self.d_k)
        self.W_v = Linear(d_model, self.num_heads * self.d_v)
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
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
            self.register_buffer("causal_mask", mask, persistent=False)
        return self.causal_mask[:seq_len, :seq_len]

    def forward(self, x: torch.Tensor, apply_rope=True, token_positions=None):
        B, S, _ = x.shape
        
        # 1. Compute Q, K, V projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Reshape to separate heads
        Q = rearrange(Q, "batch seq (h d) -> batch seq h d", h=self.num_heads)
        K = rearrange(K, "batch seq (h d) -> batch seq h d", h=self.num_heads)
        V = rearrange(V, "batch seq (h d) -> batch seq h d", h=self.num_heads)

        # 3. Apply RoPE if requested
        if apply_rope:
            if token_positions is None:
                positions = torch.arange(S, device=x.device)
            else:
                positions = token_positions

            # Reshape for RoPE: (batch * num_heads, seq_len, d_k)
            Q_reshaped = rearrange(Q, "batch seq h d -> (batch h) seq d")
            K_reshaped = rearrange(K, "batch seq h d -> (batch h) seq d")
            
            # Apply RoPE
            Q_rope = self.rope(Q_reshaped, positions)
            K_rope = self.rope(K_reshaped, positions)
            
            # Reshape back: (batch, seq_len, num_heads, d_k)
            Q = rearrange(Q_rope, "(batch h) seq d -> batch seq h d", h=self.num_heads)
            K = rearrange(K_rope, "(batch h) seq d -> batch seq h d", h=self.num_heads)

        # 4. Transpose to put heads in batch dimension for attention
        Q = rearrange(Q, "batch seq h d -> batch h seq d")
        K = rearrange(K, "batch seq h d -> batch h seq d")
        V = rearrange(V, "batch seq h d -> batch h seq d")

        # 5. Get causal mask
        mask = self._get_causal_mask(S, x.device)

        # 6. Apply scaled dot-product attention
        attention_output = scaled_dot_product_attention(Q, K, V, mask=mask)

        # 7. Reshape back to concatenate heads
        attention_output = rearrange(attention_output, "batch h seq d -> batch seq (h d)")

        # 8. Apply output projection
        output = self.W_o(attention_output)

        return output


def test_weight_assignment_methods():
    """Test different ways of assigning weights to see which one the test expects."""
    
    d_model = 64
    num_heads = 4
    seq_len = 6
    batch_size = 1
    
    # Create test input and weights
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Create test weight matrices (matching the function signature from your paste.txt)
    d_k = d_model // num_heads
    q_proj_weight = torch.randn(num_heads * d_k, d_model)
    k_proj_weight = torch.randn(num_heads * d_k, d_model)
    v_proj_weight = torch.randn(num_heads * d_k, d_model)
    o_proj_weight = torch.randn(d_model, num_heads * d_k)
    
    print(f"Test weights shapes:")
    print(f"  q_proj_weight: {q_proj_weight.shape}")
    print(f"  k_proj_weight: {k_proj_weight.shape}")
    print(f"  v_proj_weight: {v_proj_weight.shape}")
    print(f"  o_proj_weight: {o_proj_weight.shape}")
    
    # Method 1: Using .data assignment (what you want to use)
    print(f"\n=== Method 1: Using .data assignment ===")
    mhsa1 = MultiHeadSelfAttention(d_model, num_heads)
    
    print(f"Before assignment - W_q.weight.shape: {mhsa1.W_q.weight.shape}")
    print(f"Before assignment - W_o.weight.shape: {mhsa1.W_o.weight.shape}")
    
    # Check if shapes match
    print(f"q_proj_weight matches W_q.weight? {q_proj_weight.shape == mhsa1.W_q.weight.shape}")
    print(f"o_proj_weight matches W_o.weight? {o_proj_weight.shape == mhsa1.W_o.weight.shape}")
    
    if q_proj_weight.shape == mhsa1.W_q.weight.shape:
        mhsa1.W_q.weight.data = q_proj_weight
        mhsa1.W_k.weight.data = k_proj_weight
        mhsa1.W_v.weight.data = v_proj_weight
        mhsa1.W_o.weight.data = o_proj_weight
        
        output1 = mhsa1(x, apply_rope=False)
        print(f"✓ Method 1 successful: {output1.shape}, mean={output1.mean():.6f}")
    else:
        print("✗ Method 1 failed: Shape mismatch")
    
    # Method 2: Check if we need to transpose the weights
    print(f"\n=== Method 2: Try transposing weights ===")
    mhsa2 = MultiHeadSelfAttention(d_model, num_heads)
    
    try:
        # Maybe the test provides weights in transposed form
        mhsa2.W_q.weight.data = q_proj_weight.T if q_proj_weight.shape != mhsa2.W_q.weight.shape else q_proj_weight
        mhsa2.W_k.weight.data = k_proj_weight.T if k_proj_weight.shape != mhsa2.W_k.weight.shape else k_proj_weight
        mhsa2.W_v.weight.data = v_proj_weight.T if v_proj_weight.shape != mhsa2.W_v.weight.shape else v_proj_weight
        mhsa2.W_o.weight.data = o_proj_weight.T if o_proj_weight.shape != mhsa2.W_o.weight.shape else o_proj_weight
        
        output2 = mhsa2(x, apply_rope=False)
        print(f"✓ Method 2 successful: {output2.shape}, mean={output2.mean():.6f}")
        
        if 'output1' in locals():
            diff = torch.abs(output1 - output2).max()
            print(f"  Difference from Method 1: {diff:.8f}")
            
    except Exception as e:
        print(f"✗ Method 2 failed: {e}")
    
    # Method 3: Test the actual function signatures from your paste.txt
    print(f"\n=== Method 3: Test run_multihead_self_attention function ===")
    
    def run_multihead_self_attention(d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features):
        mhsa = MultiHeadSelfAttention(d_model=d_model, num_heads=num_heads)
        
        # Try the .data assignment as you planned
        mhsa.W_q.weight.data = q_proj_weight
        mhsa.W_k.weight.data = k_proj_weight
        mhsa.W_v.weight.data = v_proj_weight
        mhsa.W_o.weight.data = o_proj_weight
        
        output = mhsa.forward(in_features, apply_rope=False)
        return output
    
    try:
        output3 = run_multihead_self_attention(d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, x)
        print(f"✓ Function test successful: {output3.shape}, mean={output3.mean():.6f}")
    except Exception as e:
        print(f"✗ Function test failed: {e}")
        print(f"Expected shapes for Linear layers:")
        print(f"  W_q should be: {d_model} -> {num_heads * d_k} = weight shape {(num_heads * d_k, d_model)}")
        print(f"  W_o should be: {num_heads * d_k} -> {d_model} = weight shape {(d_model, num_heads * d_k)}")


def debug_your_actual_implementation():
    """Test with your exact MultiHeadSelfAttention class to see what's happening."""
    
    print(f"\n{'='*60}")
    print("DEBUGGING YOUR EXACT IMPLEMENTATION")
    print(f"{'='*60}")
    
    # Test with the exact shapes that your failing test probably uses
    test_cases = [
        {"d_model": 512, "num_heads": 8, "seq_len": 12, "batch_size": 2},
        {"d_model": 256, "num_heads": 4, "seq_len": 16, "batch_size": 1},
    ]
    
    for i, config in enumerate(test_cases):
        print(f"\nTest case {i+1}: {config}")
        
        d_model = config["d_model"]
        num_heads = config["num_heads"]
        seq_len = config["seq_len"]
        batch_size = config["batch_size"]
        
        # Create your MultiHeadSelfAttention
        mhsa = MultiHeadSelfAttention(d_model, num_heads)
        
        print(f"Created MHSA with:")
        print(f"  W_q.weight: {mhsa.W_q.weight.shape}")
        print(f"  W_k.weight: {mhsa.W_k.weight.shape}")
        print(f"  W_v.weight: {mhsa.W_v.weight.shape}")
        print(f"  W_o.weight: {mhsa.W_o.weight.shape}")
        
        # Test input
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Test without RoPE
        try:
            output_no_rope = mhsa(x, apply_rope=False)
            print(f"✓ No RoPE: {output_no_rope.shape}, mean={output_no_rope.mean():.6f}, std={output_no_rope.std():.6f}")
        except Exception as e:
            print(f"✗ No RoPE failed: {e}")
        
        # Test with RoPE 
        try:
            output_rope = mhsa(x, apply_rope=True)
            print(f"✓ With RoPE: {output_rope.shape}, mean={output_rope.mean():.6f}, std={output_rope.std():.6f}")
        except Exception as e:
            print(f"✗ With RoPE failed: {e}")


if __name__ == "__main__":
    test_weight_assignment_methods()
    debug_your_actual_implementation()