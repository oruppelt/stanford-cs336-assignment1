import torch
import torch.nn as nn

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

def test_final_implementation():
    theta = 10000.0
    d_k = 4
    max_seq_len = 3
    
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0]]], dtype=torch.float32)
    token_positions = torch.tensor([[0, 1]])
    
    print("Testing final implementation:")
    rope = RotaryPositionalEmbedding(theta, d_k, max_seq_len)
    result = rope(x, token_positions)
    print(f"Result: {result}")
    
    # Debug the frequency calculation
    i_indices = torch.arange(0, d_k, 2, dtype=torch.float32)
    freqs = 1.0 / (theta ** (i_indices / d_k))
    print(f"i_indices: {i_indices}")
    print(f"freqs: {freqs}")
    
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)
    print(f"angles matrix:\n{angles}")

if __name__ == "__main__":
    test_final_implementation()