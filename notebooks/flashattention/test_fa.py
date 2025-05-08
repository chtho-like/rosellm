import torch
import unittest

# Import the solve function from wherever it's defined
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from notebooks.flashattention.fa import solve

# Standard attention implementation for reference
def standard_attention(Q, K, V, d_model, h):
    N = Q.shape[0]
    dk = d_model // h
    
    # Reshape to multi-head format
    q = Q.view(N, h, dk).transpose(0, 1)
    k = K.view(N, h, dk).transpose(0, 1)
    v = V.view(N, h, dk).transpose(0, 1)
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / (dk ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v)
    
    # Reshape back
    out = out.transpose(0, 1).contiguous().view(N, d_model)
    return out

class TestFlashAttention(unittest.TestCase):
    def assert_close(self, a, b, rtol=1e-5, atol=1e-5):
        """Assert that two tensors are close within tolerance"""
        assert torch.allclose(a, b, rtol=rtol, atol=atol), \
            f"Max difference: {torch.max(torch.abs(a - b)).item()}"
    
    def test_simple_case(self):
        """Test with simple 2x4 input, 2 heads"""
        N, d_model, h = 2, 4, 2
        
        # Initialize with controlled data
        Q = torch.tensor([1.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], device="cuda").reshape(N, d_model)
        K = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], device="cuda").reshape(N, d_model)
        V = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], device="cuda").reshape(N, d_model)
        
        # Expected output from standard attention
        expected = standard_attention(Q, K, V, d_model, h)
        
        # Output from flash attention
        output = torch.zeros_like(Q)
        solve(Q, K, V, output, N, d_model, h)
        
        # Assert outputs are close
        self.assert_close(output, expected)
        print(f"Test case 1 passed. Output: {output}")
    
    def test_larger_case(self):
        """Test with larger 10x8 input, 4 heads"""
        N, d_model, h = 10, 8, 4
        
        # Initialize with random data
        torch.manual_seed(42)
        Q = torch.randn(N, d_model, device="cuda")
        K = torch.randn(N, d_model, device="cuda")
        V = torch.randn(N, d_model, device="cuda")
        
        # Expected output from standard attention
        expected = standard_attention(Q, K, V, d_model, h)
        
        # Output from flash attention
        output = torch.zeros_like(Q)
        solve(Q, K, V, output, N, d_model, h)
        
        # Assert outputs are close
        self.assert_close(output, expected)
        print(f"Test case 2 passed.")
    
    def test_single_token(self):
        """Test with single token input (N=1)"""
        N, d_model, h = 1, 4, 2
        
        Q = torch.tensor([0.5, 1.5, 2.5, 3.5], device="cuda").reshape(N, d_model)
        K = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda").reshape(N, d_model)
        V = torch.tensor([0.1, 0.2, 0.3, 0.4], device="cuda").reshape(N, d_model)
        
        # Expected output from standard attention
        expected = standard_attention(Q, K, V, d_model, h)
        
        # Output from flash attention
        output = torch.zeros_like(Q)
        solve(Q, K, V, output, N, d_model, h)
        
        # Assert outputs are close
        self.assert_close(output, expected)
        print(f"Test case 3 passed. Output: {output}")
    
    def test_odd_dimensions(self):
        """Test with dimensions that aren't multiples of block size"""
        N, d_model, h = 17, 24, 3
        
        # Initialize with random data
        torch.manual_seed(43)
        Q = torch.randn(N, d_model, device="cuda")
        K = torch.randn(N, d_model, device="cuda")
        V = torch.randn(N, d_model, device="cuda")
        
        # Expected output from standard attention
        expected = standard_attention(Q, K, V, d_model, h)
        
        # Output from flash attention
        output = torch.zeros_like(Q)
        solve(Q, K, V, output, N, d_model, h)
        
        # Assert outputs are close
        self.assert_close(output, expected)
        print(f"Test case 4 passed.")
    
    def test_many_heads(self):
        """Test with many heads (8 heads)"""
        N, d_model, h = 8, 64, 8
        
        # Initialize with random data
        torch.manual_seed(44)
        Q = torch.randn(N, d_model, device="cuda")
        K = torch.randn(N, d_model, device="cuda")
        V = torch.randn(N, d_model, device="cuda")
        
        # Expected output from standard attention
        expected = standard_attention(Q, K, V, d_model, h)
        
        # Output from flash attention
        output = torch.zeros_like(Q)
        solve(Q, K, V, output, N, d_model, h)
        
        # Assert outputs are close
        self.assert_close(output, expected, rtol=1e-4)  # Slight relaxation for numerical differences
        print(f"Test case 5 passed.")

if __name__ == "__main__":
    unittest.main()
