from typing import Any, Dict, List
import torch
import math
from auto_diff import *

class MatMulLayerNormOp(Op):
    """Fused matrix multiplication and layer normalization operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        normalized_shape: List[int], 
        eps: float = 1e-5
    ) -> Node:
        """
        Args:
            node_A: The first input node.
            node_B: The second input node.
            normalized_shape: The shape of the normalization axes.
            eps: The epsilon value to avoid division by zero.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "normalized_shape": normalized_shape,
                "eps": eps
            },
            name=f"MatMulLayerNorm({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and layer normalization result."""
        assert len(input_values) == 2
        x = input_values[0] @ input_values[1]
        indices = tuple(range(-len(node.normalized_shape), 0))
        return (x - torch.mean(x, indices, keepdim=True))/torch.sqrt(torch.var(x, indices, keepdim=True, unbiased=False) + node.eps)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
    
        dims = tuple(range(-len(node.normalized_shape), 0))
        A = node.inputs[0]
        B = node.inputs[1]
        C = matmul(A, B)
        # Compute gradient of loss w.r.t layernorm
        mean_c = mean(C, dim=dims, keepdim=True)
        var_c_eps = var_op(C, dim=dims, keepdim=True) + node.eps
        c_minus_mean = C - mean_c
        term1 = mean(output_grad, dim=dims, keepdim=True)
        term2 = mean(output_grad * (c_minus_mean), dim=dims, keepdim=True)
        inv_std = power(sqrt(var_c_eps), -1.0)
        grad_layernorm = inv_std * (output_grad - term1 - (c_minus_mean) * (term2 / (var_c_eps)))
        # Compute gradient of loss w.r.t matmullayernorm
        grad_1 = matmul(grad_layernorm, transpose(B, -2, -1))
        grad_2 = matmul(transpose(A, -2, -1), grad_layernorm)

        return [grad_1, grad_2]



class MatMulSoftmaxOp(Op):
    """Fused matrix multiplication and softmax operation."""

    def __call__(
        self, 
        node_A: Node, 
        node_B: Node, 
        dim: int = -1
    ) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            attrs={
                "dim": dim
            },
            name=f"MatMulSoftmax({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the fused matmul and softmax result."""
        assert len(input_values) == 2
        x = input_values[0] @ input_values[1]
        # Softmax is shift invariant, subtracting max_val numerically stabilizes it without changing output
        max_val = torch.max(x, dim=node.dim, keepdim=True)[0]
        e = torch.exp(x - max_val)
        return e/torch.sum(e, dim=node.dim, keepdim=True)
        

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of fused node, return partial adjoints to each input."""
        # First compute the forward pass result we need for softmax gradient
        A = node.inputs[0]
        B = node.inputs[1]
        y = matmul_softmax(A, B, dim=node.dim)
        dot = sum_op(y * output_grad, node.dim, keepdim=True)
        grad_softmax = y * (output_grad - dot)
        grad_1 = matmul(grad_softmax, transpose(B, -2, -1))
        grad_2 = matmul(transpose(A, -2, -1), grad_softmax)
        return [grad_1, grad_2]


# Create global instances of the fused ops
matmul_layernorm = MatMulLayerNormOp()
matmul_softmax = MatMulSoftmaxOp()