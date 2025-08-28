import torch
import torch.nn as nn
import math
from einops import einsum

# Version 1: Your original (in_features, out_features)
class LinearV1(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features, device=device))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

# Version 2: Corrected (out_features, in_features)  
class LinearV2(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

def test_both_versions():
    d_in, d_out = 3, 4
    
    # Create test weights (d_out, d_in) as provided by test
    test_weights = torch.tensor([
        [1.0, 2.0, 3.0],    # Row 1 of weight matrix
        [4.0, 5.0, 6.0],    # Row 2 
        [7.0, 8.0, 9.0],    # Row 3
        [10.0, 11.0, 12.0]  # Row 4
    ])  # Shape: (4, 3) = (d_out, d_in)
    
    print(f"Test weights shape: {test_weights.shape}")
    print(f"Test weights:\n{test_weights}")
    
    # Test input
    x = torch.tensor([[1.0, 0.0, 1.0]])  # Shape: (1, 3) = (1, d_in)
    print(f"\nInput x: {x}")
    
    print("\n" + "="*50)
    print("VERSION 1: Original (in_features, out_features)")
    print("="*50)
    
    linear_v1 = LinearV1(d_in, d_out)
    print(f"V1 initial weight shape: {linear_v1.weight.shape}")
    
    linear_v1.weight.data = test_weights
    print(f"V1 weight shape after assignment: {linear_v1.weight.shape}")
    print(f"V1 weight after assignment:\n{linear_v1.weight.data}")
    
    try:
        output_v1 = linear_v1(x)
        print(f"V1 output: {output_v1}")
    except Exception as e:
        print(f"V1 failed: {e}")
    
    print("\n" + "="*50)
    print("VERSION 2: Corrected (out_features, in_features)")
    print("="*50)
    
    linear_v2 = LinearV2(d_in, d_out)
    print(f"V2 initial weight shape: {linear_v2.weight.shape}")
    
    linear_v2.weight.data = test_weights
    print(f"V2 weight shape after assignment: {linear_v2.weight.shape}")
    print(f"V2 weight after assignment:\n{linear_v2.weight.data}")
    
    try:
        output_v2 = linear_v2(x)
        print(f"V2 output: {output_v2}")
    except Exception as e:
        print(f"V2 failed: {e}")
    
    print("\n" + "="*50)
    print("MANUAL CALCULATION")
    print("="*50)
    
    # What should the correct answer be?
    # Standard matrix multiplication: x @ W.T
    # Where W is the test_weights (4, 3), so W.T is (3, 4)
    expected = x @ test_weights.T
    print(f"Expected result (x @ weights.T): {expected}")
    
    # Check which version gives the correct mathematical result
    print(f"\nV1 matches expected? {torch.allclose(output_v1, expected) if 'output_v1' in locals() else 'N/A'}")
    print(f"V2 matches expected? {torch.allclose(output_v2, expected) if 'output_v2' in locals() else 'N/A'}")

if __name__ == "__main__":
    test_both_versions()