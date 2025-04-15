from typing import Any, Dict, List

import torch
import math


class Node:
    """Node in a computational graph.

    Fields
    ------
    inputs: List[Node]
        The list of input nodes to this node.

    op: Op
        The op of this node.

    attrs: Dict[str, Any]
        The attribute dictionary of this node.
        E.g. "constant" is the constant operand of add_by_const.

    name: str
        Name of the node for debugging purposes.
    """

    inputs: List["Node"]
    op: "Op"
    attrs: Dict[str, Any]
    name: str

    def __init__(
        self, inputs: List["Node"], op: "Op", attrs: Dict[str, Any] = {}, name: str = ""
    ) -> None:
        self.inputs = inputs
        self.op = op
        self.attrs = attrs
        self.name = name

    def __add__(self, other):
        if isinstance(other, Node):
            return add(self, other)
        else:
            assert isinstance(other, (int, float))
            return add_by_const(self, other)

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __mul__(self, other):
        if isinstance(other, Node):
            return mul(self, other)
        else:
            assert isinstance(other, (int, float))
            return mul_by_const(self, other)

    def __truediv__(self, other):
        if isinstance(other, Node):
            return div(self, other)
        else:
            assert isinstance(other, (int, float))
            return div_by_const(self, other)
    
    def __rtruediv__(self, other):
        if isinstance(other, Node):
            return div(other, self)
        else:
            assert isinstance(other, (int, float))
            return mul_by_const(power(self, -1.0), other)
        
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        return power(self, other)

    # Allow left-hand-side add and multiplication.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow printing the node name."""
        return self.name

    def __getattr__(self, attr_name: str) -> Any:
        if attr_name in self.attrs:
            return self.attrs[attr_name]
        raise KeyError(f"Attribute {attr_name} does not exist in node {self}")

    __repr__ = __str__


class Variable(Node):
    """A variable node with given name."""

    def __init__(self, name: str) -> None:
        super().__init__(inputs=[], op=placeholder, name=name)


class Op:
    """The class of operations performed on nodes."""

    def __call__(self, *kwargs) -> Node:
        """Create a new node with this current op.

        Returns
        -------
        The created new node.
        """
        raise NotImplementedError

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Compute the output value of the given node with its input
        node values given.

        Parameters
        ----------
        node: Node
            The node whose value is to be computed

        input_values: List[torch.Tensor]
            The input values of the given node.

        Returns
        -------
        output: torch.Tensor
            The computed output value of the node.
        """
        raise NotImplementedError

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given a node and its output gradient node, compute partial
        adjoints with regards to each input node.

        Parameters
        ----------
        node: Node
            The node whose inputs' partial adjoints are to be computed.

        output_grad: Node
            The output gradient with regard to given node.

        Returns
        -------
        input_grads: List[Node]
            The list of partial gradients with regard to each input of the node.
        """
        raise NotImplementedError


class PlaceholderOp(Op):
    """The placeholder op to denote computational graph input nodes."""

    def __call__(self, name: str) -> Node:
        return Node(inputs=[], op=self, name=name)

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        raise RuntimeError(
            "Placeholder nodes have no inputs, and their values cannot be computed."
        )

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        raise RuntimeError("Placeholder nodes have no inputs.")


class AddOp(Op):
    """Op to element-wise add two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}+{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of input values."""
        assert len(input_values) == 2
        return input_values[0] + input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to each input."""
        return [output_grad, output_grad]


class AddByConstOp(Op):
    """Op to element-wise add a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}+{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise addition of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] + node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of add node, return partial adjoint to the input."""
        return [output_grad]


class MulOp(Op):
    """Op to element-wise multiply two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}*{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of input values."""
        assert len(input_values) == 2
        return input_values[0] * input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to each input."""
        return [output_grad * node.inputs[1], output_grad * node.inputs[0]]
    

class MulByConstOp(Op):
    """Op to element-wise multiply a node by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}*{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise multiplication of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] * node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of multiplication node, return partial adjoint to the input."""
        return [output_grad * node.constant]
    
class GreaterThanOp(Op):
    """Op to compare if node_A > node_B element-wise."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}>{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return element-wise comparison result as float tensor."""
        assert len(input_values) == 2
        return (input_values[0] > input_values[1]).float().to(input_values[0].dtype)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Comparison operations have gradient of 0."""
        return [zeros_like(node.inputs[0]), zeros_like(node.inputs[1])]
    
class EqualOp(Op):
    """Op to compare if node_A = node_B element-wise."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}={node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return element-wise comparison result as float tensor."""
        assert len(input_values) == 2
        return (input_values[0] == input_values[1]).float().to(input_values[0].dtype)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Comparison operations have gradient of 0."""
        return [zeros_like(node.inputs[0]), zeros_like(node.inputs[1])]

class SubOp(Op):
    """Op to element-wise subtract two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}-{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise subtraction of input values."""
        assert len(input_values) == 2
        return input_values[0] - input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of subtraction node, return partial adjoint to each input."""
        return [output_grad, mul_by_const(output_grad, -1)]
    
class ZerosLikeOp(Op):
    """Zeros-like op that returns an all-zero array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"ZerosLike({node_A.name})")

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return an all-zero tensor with the same shape as input."""
        assert len(input_values) == 1
        return torch.zeros_like(input_values[0], dtype=input_values[0].dtype)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]

class OnesLikeOp(Op):
    """Ones-like op that returns an all-one array with the same shape as the input."""

    def __call__(self, node_A: Node) -> Node:
        return Node(inputs=[node_A], op=self, name=f"OnesLike({node_A.name})")

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return an all-one tensor with the same shape as input."""
        assert len(input_values) == 1
        return torch.ones_like(input_values[0], dtype=input_values[0].dtype)   

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [zeros_like(node.inputs[0])]

class SumOp(Op):
    """
    Op to compute sum along specified dimensions.
    
    Note: This is a reference implementation for SumOp.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, dim: tuple, keepdim: bool = False) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Sum({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return input_values[0].sum(dim=node.dim, keepdim=node.keepdim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        x = node.inputs[0]
        if node.keepdim:
            return [expand_as(output_grad, x)]
        else:
            unsqueeze_dims = sorted(list(node.dim))
            reshape_grad = output_grad
            for i in unsqueeze_dims:
                reshape_grad = unsqueeze_op(reshape_grad, i)
            return [expand_as(reshape_grad, x)]

class ExpandAsOp(Op):
    """Op to broadcast a tensor to the shape of another tensor.
    
    Note: This is a reference implementation for ExpandAsOp.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast({node_A.name} -> {node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values
        return input_tensor.expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the broadcast node, compute partial adjoint to input."""
        """TODO: FIX THIS"""
        return [sum_op(output_grad,dim=0), zeros_like(output_grad)]
    
class ExpandAsOp3d(Op):
    """Op to broadcast a tensor to the shape of another tensor.
    
    Note: This is a reference implementation for ExpandAsOp3d.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"broadcast({node_A.name} -> {node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 2
        input_tensor, target_tensor = input_values
        return input_tensor.unsqueeze(1).expand_as(target_tensor)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the broadcast node, compute partial adjoint to input."""
        
        return [sum_op(output_grad,dim=(0, 1)), zeros_like(output_grad)]

class LogOp(Op):
    """Logarithm (natural log) operation."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"Log({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the natural logarithm of the input."""
        assert len(input_values) == 1, "Log operation requires one input."
        return torch.log(input_values[0])

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given the gradient of the Log node, return the partial adjoint to the input."""
        input_node = node.inputs[0]
        return [output_grad / input_node]


class BroadcastOp(Op):
    def __call__(self, node_A: Node, input_shape: List[int], target_shape: List[int]) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"input_shape": input_shape, "target_shape": target_shape},
            name=f"Broadcast({node_A.name}, {target_shape})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the broadcasted tensor."""
        assert len(input_values) == 1
        return input_values[0].expand(node.target_shape)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of broadcast node, return partial adjoint to input.
        
        For broadcasting, we need to sum out the broadcasted dimensions to get
        back to the original shape.
        """
        if "input_shape" not in node.attrs:
            raise ValueError("Input shape is not set. Make sure compute() is called before gradient()")
            
        input_shape = node.input_shape
        output_shape = node.target_shape
        
        dims_to_sum = []
        for i, (in_size, out_size) in enumerate(zip(input_shape[::-1], output_shape[::-1])):
            if in_size != out_size:
                dims_to_sum.append(len(output_shape) - 1 - i)
                
        grad = output_grad
        if dims_to_sum:
            grad = sum_op(grad, dim=dims_to_sum, keepdim=True)
            
        if len(output_shape) > len(input_shape):
            grad = sum_op(grad, dim=list(range(len(output_shape) - len(input_shape))), keepdim=False)
            
        return [grad]

class DivOp(Op):
    """Op to element-wise divide two nodes."""

    def __call__(self, node_A: Node, node_B: Node) -> Node:
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}/{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of input values."""
        assert len(input_values) == 2
        return input_values[0]/input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to each input."""
        grad_1 = output_grad / node.inputs[1]
        grad_2 = -1 * output_grad * node.inputs[0] / pow(node.inputs[1], 2)
        return [grad_1, grad_2]

class DivByConstOp(Op):
    """Op to element-wise divide a nodes by a constant."""

    def __call__(self, node_A: Node, const_val: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"constant": const_val},
            name=f"({node_A.name}/{const_val})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the element-wise division of the input value and the constant."""
        assert len(input_values) == 1
        return input_values[0] / node.constant

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of division node, return partial adjoint to the input."""
        return [output_grad / node.constant]

class TransposeOp(Op):
    """Op to transpose a matrix."""

    def __call__(self, node_A: Node, dim0: int, dim1: int) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim0": dim0, "dim1": dim1},
            name=f"transpose({node_A.name}, {dim0}, {dim1})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the transpose of the input by swapping two dimensions.
        
        For example:
        - transpose(x, 1, 0) swaps first two dimensions
        """
        assert len(input_values) == 1
        return input_values[0].transpose(node.dim0, node.dim1)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of transpose node, return partial adjoint to input."""
        return [transpose(output_grad, node.dim0, node.dim1)]

class MatMulOp(Op):
    """Matrix multiplication op of two nodes."""

    def __call__(
        self, node_A: Node, node_B: Node
    ) -> Node:
        """Create a matrix multiplication node.

        Parameters
        ----------
        node_A: Node
            The lhs matrix.
        node_B: Node
            The rhs matrix

        Returns
        -------
        result: Node
            The node of the matrix multiplication.
        """
        return Node(
            inputs=[node_A, node_B],
            op=self,
            name=f"({node_A.name}@{node_B.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return the matrix multiplication result of input values."""
        assert len(input_values) == 2
        return input_values[0] @ input_values[1]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of matmul node, return partial adjoint to each input."""
        grad_1 = matmul(output_grad, transpose(node.inputs[1], -2, -1))
        grad_2 = matmul(transpose(node.inputs[0], -2, -1), output_grad)
        return [grad_1, grad_2]
    

class SoftmaxOp(Op):
    """Softmax operation on input node."""

    def __call__(self, node_A: Node, dim: int = -1) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim},
            name=f"Softmax({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return softmax of input along specified dimension."""
        assert len(input_values) == 1
        # Softmax is shift invariant, subtracting max_val numerically stabilizes it without changing output
        max_val = torch.max(input_values[0], dim=node.dim, keepdim=True)[0]
        e = torch.exp(input_values[0] - max_val)
        return e/torch.sum(e, dim=node.dim, keepdim=True)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of softmax node, return partial adjoint to input."""
        y = softmax(node.inputs[0])
        dot = sum_op(y * output_grad, node.dim, keepdim=True)
        return [y * (output_grad - dot)]
    
class MaxOp(Op):
    """return the max element in X along a dimension d"""

    def __call__(self, node_A: Node, dim: int = -1, keepdim: bool = False) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Max({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return max element(s) of input along specified dimension."""
        assert len(input_values) == 1
        return torch.max(input_values[0], dim=node.dim, keepdim=node.keepdim)[0]

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of softmax node, return partial adjoint to input."""
        a = node.inputs[0]
        max_mask = eq_op(a, max_op(a, dim=node.dim, keepdim=True))
        grad = max_mask / sum_op(max_mask, dim=(node.dim,), keepdim=True)
        if node.keepdim:
            return [output_grad * grad]
        else:
            return [unsqueeze_op(output_grad, dim=node.dim) * grad]
        

class LayerNormOp(Op):
    """Layer normalization operation."""

    def __call__(self, node_A: Node, normalized_shape: List[int], eps: float = 1e-5) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"normalized_shape": normalized_shape, "eps": eps},
            name=f"LayerNorm({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return layer normalized input."""
        assert len(input_values) == 1
        x = input_values[0]
        indices = tuple(range(-len(node.normalized_shape), 0))
        return (x - torch.mean(x, indices, keepdim=True))/torch.sqrt(torch.var(x, indices, keepdim=True, unbiased=False) + node.eps)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """
        Gradient of the LayerNorm node wrt input x
        (assuming no gamma/beta for simplicity).
        """
        dims = tuple(range(-len(node.normalized_shape), 0))
        
        x = node.inputs[0]
        mean_x = mean(x, dim=dims, keepdim=True)
        var_x_eps = var_op(x, dim=dims, keepdim=True) + node.eps
        x_minus_mean = x - mean_x

        # sum1 = average of output_grad across LN dims
        term1 = mean(output_grad, dim=dims, keepdim=True)
        # sum2 = average of (output_grad * (x-mean))
        term2 = mean(output_grad * (x_minus_mean), dim=dims, keepdim=True)
        # inv_std = 1 / sqrt(var + eps)
        inv_std = power(sqrt(var_x_eps), -1.0)
        # final LN gradient wrt x
        grad_x = inv_std * (output_grad - term1 - (x_minus_mean) * (term2 / (var_x_eps)))

        return [grad_x]


class ReLUOp(Op):
    """ReLU activation function."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"ReLU({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        """Return ReLU of input."""
        assert len(input_values) == 1
        return torch.maximum(input_values[0], torch.zeros_like(input_values[0]))

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        """Given gradient of ReLU node, return partial adjoint to input."""
        mask = greater(node.inputs[0], zeros_like(node.inputs[0]))
        return [mask * output_grad]


class SqrtOp(Op):
    """Op to compute element-wise square root."""

    def __call__(self, node_A: Node) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            name=f"Sqrt({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return input_values[0].sqrt()

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [0.5 * (ones_like(node.inputs[0])/(sqrt(node.inputs[0]))) * output_grad]

class PowerOp(Op):
    """Op to compute element-wise power."""

    def __call__(self, node_A: Node, exponent: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"exponent": exponent},
            name=f"Power({node_A.name}, {exponent})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return input_values[0].pow(node.exponent)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        n = node.attrs['exponent']
        return [output_grad * n * (node.inputs[0] ** (n-1))]
    
class UnsqueezeOp(Op):
    """Op to add a singleton dimension at a given dimensional index of a node"""

    def __call__(self, node_A: Node, dim: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim},
            name=f"Unsqueeze({node_A.name}, at dim {dim})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return input_values[0].unsqueeze(node.dim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [squeeze_op(output_grad, node.dim)]

class SqueezeOp(Op):
    """Op to remove a singleton dimension at a given dimensional index of a node"""

    def __call__(self, node_A: Node, dim: float) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim},
            name=f"Squeeze({node_A.name}, at dim {dim})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return input_values[0].squeeze(node.dim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        return [unsqueeze_op(output_grad, node.dim)]

class MeanOp(Op):
    """Op to compute mean along specified dimensions.
    
    Note: This is a reference implementation for MeanOp.
        If it does not work in your case, you can modify it.
    """

    def __call__(self, node_A: Node, dim: tuple, keepdim: bool = False) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Mean({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.mean(input_values[0], dim=node.dim, keepdim=node.keepdim)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        x = node.inputs[0]
        D = expand_as(sum_op(ones_like(x), dim=node.dim, keepdim=True), x)

        if node.keepdim:
            return [div(output_grad, D)]
        else:
            unsqueeze_dims = sorted(list(node.dim))
            reshape_grad = output_grad
            for i in unsqueeze_dims:
                reshape_grad = unsqueeze_op(reshape_grad, i)
            return [div(reshape_grad, D)]

class VarianceOp(Op):
    """Op to compute variance along specified dimensions. Unbiased Only"""

    def __call__(self, node_A: Node, dim: tuple, keepdim: bool = False) -> Node:
        return Node(
            inputs=[node_A],
            op=self,
            attrs={"dim": dim, "keepdim": keepdim},
            name=f"Variance({node_A.name})",
        )

    def compute(self, node: Node, input_values: List[torch.Tensor]) -> torch.Tensor:
        assert len(input_values) == 1
        return torch.var(input_values[0], dim=node.dim, keepdim=node.keepdim, unbiased=False)

    def gradient(self, node: Node, output_grad: Node) -> List[Node]:
        x = node.inputs[0]
        mean_x = mean(x, dim=node.dim, keepdim=True)
        D = expand_as(sum_op(ones_like(x), dim=node.dim, keepdim=True), x)
        
        if node.keepdim:
            return [div(2 * output_grad * (x-mean_x), D)]
        else:
            unsqueeze_dims = sorted(list(node.dim))
            reshape_grad = output_grad
            for i in unsqueeze_dims:
                reshape_grad = unsqueeze_op(reshape_grad, i)
            return [div(2 * reshape_grad * (x-mean_x), D)]

# Create global instances of ops.
# Your implementation should just use these instances, rather than creating new instances.
placeholder = PlaceholderOp()
add = AddOp()
mul = MulOp()
div = DivOp()
add_by_const = AddByConstOp()
mul_by_const = MulByConstOp()
div_by_const = DivByConstOp()
matmul = MatMulOp()
zeros_like = ZerosLikeOp()
ones_like = OnesLikeOp()
softmax = SoftmaxOp()
layernorm = LayerNormOp()
relu = ReLUOp()
transpose = TransposeOp()
mean = MeanOp()
sum_op = SumOp()
sqrt = SqrtOp()
power = PowerOp()
greater = GreaterThanOp()
max_op = MaxOp()
eq_op = EqualOp()
expand_as = ExpandAsOp()
expand_as_3d = ExpandAsOp3d()
log = LogOp()
sub = SubOp()
broadcast = BroadcastOp()
var_op = VarianceOp()
unsqueeze_op = UnsqueezeOp()
squeeze_op = SqueezeOp()

def topological_sort(nodes):
    """Helper function to perform topological sort on nodes.
    
    Parameters
    ----------
    nodes : List[Node] or Node
        Node(s) to sort
        
    Returns
    -------
    List[Node]
        Nodes in topological order
    """
    visited = set()
    order = []
    def dfs(node):
        if node in visited:
            return

        for parent in node.inputs:
            if parent not in visited:
                dfs(parent)
        
        order.append(node)
        visited.add(node)
        

    for node in nodes:
        if node not in visited:
            dfs(node)

    return order

class Evaluator:
    """The node evaluator that computes the values of nodes in a computational graph."""

    eval_nodes: List[Node]

    def __init__(self, eval_nodes: List[Node]) -> None:
        """Constructor, which takes the list of nodes to evaluate in the computational graph.

        Parameters
        ----------
        eval_nodes: List[Node]
            The list of nodes whose values are to be computed.
        """
        self.eval_nodes = eval_nodes

    def run(self, input_values: Dict[Node, torch.Tensor]) -> List[torch.Tensor]:
        """Computes values of nodes in `eval_nodes` field with
        the computational graph input values given by the `input_values` dict.

        Parameters
        ----------
        input_values: Dict[Node, torch.Tensor]
            The dictionary providing the values for input nodes of the
            computational graph.
            Throw ValueError when the value of any needed input node is
            not given in the dictionary.

        Returns
        -------
        eval_values: List[torch.Tensor]
            The list of values for nodes in `eval_nodes` field.
        """
        ordered = topological_sort(self.eval_nodes)
        for node in ordered:
            if not node.inputs:
                continue

            inputs = []
            for n in node.inputs:
                if n not in input_values:
                    raise ValueError("Needed input node is not given in the dictionary")
                inputs.append(input_values[n])

            input_values[node] = node.op.compute(node, inputs)
        
        return [input_values[n] for n in self.eval_nodes]

def gradients(output_node: Node, nodes: List[Node]) -> List[Node]:
    """Construct the backward computational graph, which takes gradient
    of given output node with respect to each node in input list.
    Return the list of gradient nodes, one for each node in the input list.

    Parameters
    ----------
    output_node: Node
        The output node to take gradient of, whose gradient is 1.

    nodes: List[Node]
        The list of nodes to take gradient with regard to.

    Returns
    -------
    grad_nodes: List[Node]
        A list of gradient nodes, one for each input nodes respectively.
    """
    reverse_topo = list(reversed(topological_sort([output_node])))
    grads = {n:[] for n in reverse_topo}

    grads[output_node].append(ones_like(output_node))

    for node in reverse_topo:
        grads[node] = sum(grads[node])
        if node.inputs:
            inp_grads = node.op.gradient(node, grads[node])
            for n, g in zip(node.inputs, inp_grads):
                grads[n].append(g)
    
    return [grads[inp] if inp in grads else zeros_like(inp) for inp in nodes]
    
        

    
