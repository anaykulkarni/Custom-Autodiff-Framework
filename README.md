## **A declarative auto-differentiation framework**

Automatic differentiation forms the core technique for training machine learning models. In this assignment, you are required to develop a basic automatic differentiation system from scratch. Several functions will be given to you, and you will focus on creating standard operations used in transformers and other ML architectures - namely LayerNorm, ReLU, Softmax, among others.

Let's first go through a run-down of how autodiff works.

The automatic differentiation algorithm in this assignment operates using a computational graph. A computational graph visually represents the sequence of operations needed to compute an expression. You can reference lecture 2, slide 37 for a quick example on how this graph works.

Let's begin by exploring the fundamental concepts and data structures used in the framework. A computational graph is composed of nodes, each representing a distinct computation step in the evaluation of the entire expression.

Each node consists of three components, as shown in auto_diff.py line 6:

*   an operation (field op), specifying the type of computation the node performs.
*   a list of input nodes (field inputs), detailing the sources of input for the computation.
*   optionally, additional "attributes" (field attrs), which vary depending on the node's operation.

These attributes will be discussed in more detail later in this section.

Input nodes in a computational graph can be defined using ad.Variable. For instance, the input variable nodes $x_1$ and $x_2$ might be set up as follows:

```python
import auto_diff as ad

x1 = ad.Variable(name="x1")
x2 = ad.Variable(name="x2")
```

In auto_diff.py (line 81), the ad.Variable class is used to create a node with the operation placeholder and a specified name. Input nodes have empty inputs and attrs:

```python
class Variable(Node):
    def __init__(self, name: str) -> None:
        super().__init__(inputs=[], op=placeholder, name=name)
```

Here, the placeholder operation signifies that the input variable node does not perform any computation. Apart from placeholder, there are other operations defined in auto_diff.py, like add and matmul. You should not create your own instances of these ops.

Returning to our example where
$y = x_1 * x_2 + x_1$, with x1 and x2 already established as input variables, the rest of the graph can be defined using just one line of Python:

```python
y = x1 * x2 + x1
```

This code first creates a node with the operation mul, taking x1 and x2 as its inputs. It then constructs another node with add, which utilizes the result of the multiplication node along with x1 as inputs. Consequently, this computational graph ultimately comprises four nodes.

Important Note

It's important to note that a computational graph (e.g., the four nodes we defined) does not inherently store the actual values of its nodes. The structure of this assignment aligns with the TensorFlow v1 approach that was covered in our lectures. This method contrasts with frameworks like PyTorch, where input tensor values are specified upfront, and the values of intermediate tensors are computed immediately as they are defined.

In our framework, to calculate the value of the output y given the inputs x1 and x2, we utilize the Evaluator class found in auto_diff.py at line 373.

#### **Evaluator**

Here's a walkthrough of how Evaluator works. The constructor of Evaluator accepts a list of nodes that it needs to evaluate. By initiating an Evaluator with:

```evaluator = ad.Evaluator(eval_nodes=[y])```

you are essentially setting up an Evaluator instance designed to compute the value of y. To calculate this, input tensor values are provided via the Evaluator.run method, which you will implement. These input tensors are assumed to be of type numpy.ndarray throughout this assignment. Here’s how it works:

```python
import numpy as np

x1_value = np.array(2)
x2_value = np.array(3)
y_value = evaluator.run(input_dict={x1: x1_value, x2: x2_value})

```

In this process, the run method takes the input values using a dictionary of the form `Dict[Node, numpy.ndarray]`, calculates the value of the node y internally, and outputs the result. For instance, with the input values 2 * 3 + 2 = 8, the expected result for y_value would be `np.ndarray(8)`.

The `Evaluator.run` method is responsible for the forward computation of nodes. Building on what was discussed in the lectures, to calculate the gradient of the output with respect to each input node within a computational graph, we enhance the forward graph with an additional backward component. By integrating both forward and backward graphs, and providing values for the input nodes, the Evaluator can compute the output value, the loss value, and the gradient values for each input node in a single execution of `Evaluator.run`.

You are tasked with implementing the function ```gradients(output_node: Node, nodes: List[Node]) -> List[Node]``` for some of the operators found in auto_diff.py. This function constructs the backward graph needed for gradient computation. It accepts an output node—typically the node representing the loss function in machine learning applications, where the gradient is preset to 1. It also takes a list of nodes for which gradients are to be computed and returns a list of gradient nodes corresponding to each node in the input list.

Returning to our earlier example, once you have implemented the gradients function, you can use it to calculate the gradients of $y$ with respect to $x_1$ and $x_2$. This is done by running:

```x1_grad, x2_grad = ad.gradients(output_node=y, node=[x1, x2])```

to obtain the respective gradients. Following this, you can set up an Evaluator with nodes y, x1_grad, and x2_grad. This allows you to use the Evaluator.run method to compute both the output value and the gradients for the input nodes.

Before you start working on the assignment, let's clarify how operations (ops) work. Within auto_diff.py, each op is equipped with three methods:

```__call__(self, **kwargs) -> Node```:

*  accepts input nodes (and attributes), creates a new node utilizing this op, and returns the newly created node.

```compute(self, node: Node, input_values: List[torch.Tensor])-> torch.Tensor```

*  processes the specified node along with its input values and delivers the resultant node value.

```gradient(self, node: Node, output_grad: Node) -> List[Node]```

*  receives a node and its gradient node, returning the partial adjoint nodes for each input node.

In essence, the `Op.compute` method is responsible for calculating the value of an individual node based on its inputs, while the `Evaluator.run` function computes the value of the entire graph's output based on the graph's inputs. The `Op.gradient` method is designed to construct the backward computational graph for an individual node, whereas the gradients function builds the backward graph for the entire graph. Accordingly, your implementation of `Evaluator.run `should effectively utilize the compute method from op, and your implementation of the gradients function should make use of the gradient method provided by op.

#### **Testing**

To run tests use:

```bash
pytest tests/test_auto_diff_node_forward.py
pytest tests/test_auto_diff_node_backward.py
pytest tests/test_auto_diff_graph_forward.py
pytest tests/test_auto_diff_graph_backward.py
```

### Implementating a Vision Transformer with Custom Autodiff

This project implements the core components of a transformer model from scratch using a custom-built automatic differentiation engine. The goal is to reproduce key functionalities of the transformer architecture—such as linear projections, attention mechanisms, and encoder layers—without relying on deep learning libraries like PyTorch for forward or backward computation.

### Overview

The implementation focuses on building a simplified version of a transformer layer suitable for training on image data (e.g., MNIST). For pedagogical clarity and efficiency, this version does **not** include residual connections or multi-head attention. All computations are built on top of custom operations defined in the `auto_diff` module.

### Key Components

#### Linear Layer

A linear transformation module is implemented using:

\[
\text{output} = \text{input} @ \text{weight} + \text{bias}
\]

This forms the foundation for both projection layers and feedforward submodules.

#### Single-Head Scaled Dot-Product Attention

Implements the standard scaled dot-product attention mechanism:

\[
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
\]
\[
A = \text{Softmax} \left( \frac{QK^\top}{\sqrt{d_k}} \right),\quad \text{Output} = AV
\]

All mathematical operations are constructed using the custom autodiff graph.

#### Transformer Encoder Layer

Combines self-attention and a feedforward network into a single transformer encoder layer, chaining the previously defined components. Residual connections and layer normalization are omitted in this minimal design for conceptual clarity.

### Training and Evaluation

The file `transformer.py` includes an end-to-end training loop built around a transformer-based architecture applied to the MNIST digit classification task.

Implemented components:
- `transformer()`: Assembles the full transformer forward pass.
- `softmax_loss()`: Computes cross-entropy loss between predicted logits and true labels.
- `sgd_epoch()`: Performs a single epoch of training with stochastic gradient descent.

Run training with:

```bash
python transformer.py
```

### Expected Outcome

After training, the model is expected to reach a test accuracy of **at least 50%**, demonstrating that meaningful learning has taken place even with this simplified architecture. Due to the absence of advanced architectural features (e.g., residuals, multi-head attention), extremely high performance is not expected—but this implementation serves as a clear, educational example of how a transformer can be built from first principles.

#### BroadcastOp Deep Dive
In question 1, we implemented `BroadcastOp` for you, takes a tensor of a given shape and "expands" it so that it matches a larger, target shape, enabling element-wise operations between tensors of different but compatible dimensions.

A quick example: Suppose you have a tensor \( x \) of shape \([1, 5]\). You want to perform
an element-wise operation that requires a shape \([3, 5]\). You’d call:

```python
broadcast(x, input_shape=[1, 5], target_shape=[3, 5])
```

Forward Pass:
x is shape [1, 5] and is expanded to shape [3, 5]. )t creates a view that repeats the single row across the new dimension until the shape matches [3, 5].

Backward Pass:
During backpropagation, the gradient at shape [3,5] must be “collapsed” back to [1,5], so you sum along the newly expanded dimension(s). That way, if each of the 3 rows contributed to the gradient, those contributions are correctly aggregated into the single row of the original tensor shape.

### Fused Operations: Improving Efficiency in Transformer Computations

This project also includes implementations of fused operators that combine commonly used deep learning operations into a single computational kernel. Fused operators improve performance by reducing memory bandwidth usage, eliminating redundant reads/writes, and lowering function-call overhead. These techniques are often used in high-performance transformer architectures.

### Fused MatMul + LayerNorm

Implements a fused operation that:
- Performs matrix multiplication between inputs A and B
- Applies layer normalization across the last dimension
- Computes gradients for backpropagation

Layer normalization statistics (mean and variance) are calculated during the matrix multiplication step to reduce extra passes over the data. This avoids storing intermediate results and improves computational efficiency.

The implementation supports both 2D and 3D batched matrix inputs. When operating in 3D, each batch is normalized independently along the last dimension of the result.

### Fused MatMul + Softmax

Implements a fused operation that:
- Performs matrix multiplication between inputs A and B
- Applies the softmax operation along the last dimension
- Computes gradients for backpropagation

Softmax is computed in a numerically stable way using max-subtraction to prevent overflow. This operation is particularly useful in attention mechanisms in transformer architectures, where matrix multiplication is immediately followed by softmax.

### Benefits of Fusion

- **Reduced Memory Access:** Eliminates the need to write and read intermediate results between operations.
- **Lower Kernel Overhead:** Reduces overhead by combining computation into a single execution context.
- **Better Cache Utilization:** Keeps data in high-speed registers or local cache during execution.
- **Arithmetic Efficiency:** Computes normalization statistics during the matrix multiplication phase, reducing redundancy.

Although these fused operators are implemented using structured function calls rather than hardware-optimized kernels, they illustrate the design pattern used in real-world systems like XLA, TensorRT, or TVM.

These foundational implementations set the stage for future improvements, such as kernel tiling, vectorization, and parallel thread execution tailored for GPU/TPU acceleration.

```python
-----------------------------------------------------------------------------------------------------
## Matmul + Layer norm (handles 2D and 3D matmul with layer norm for last dimension only)
-----------------------------------------------------------------------------------------------------

# Has batch dimension?
reduce = False
if len(A.shape) == 2:
    reduce = True
    A, B = A.unsqueeze(0), B.unsqueeze(0)
    batch = 1
else:
    batch = A.shape[-3]

# Shapes of last two dimensions
M, N, P = A.shape[-2], A.shape[-1], B.shape[-1]

# output matrix
C = torch.zeros(batch, M, P).to(A.device)

mean, var = 0.0, 0.0
for b in range(batch):
    for m in range(M):
        sum = 0.0
        sum_of_square = 0.0
        for p in range(P):
            for n in range(N):
                C[b, m, p] += A[b, m, n] * B[b, n, p]
            sum += C[b, m, p]
            sum_of_square += C[b, m, p] ** 2
        mean = sum/P
        var = (sum_of_square/P) - mean**2
        C[b, m] = (C[b, m] - mean)/torch.sqrt(var + node.eps)

return C.squeeze(0) if reduce else C 

-----------------------------------------------------------------------------------------------------
## Matmul + Softmax (handles 2D and 3D matmul with softmax for last dimension only)
-----------------------------------------------------------------------------------------------------

# Has batch dimension?
reduce = False
if len(A.shape) == 2:
    reduce = True
    A, B = A.unsqueeze(0), B.unsqueeze(0)
    batch = 1
else:
    batch = A.shape[-3]

# Shapes of last two dimensions
M, N, P = A.shape[-2], A.shape[-1], B.shape[-1]

# output matrix
C = torch.zeros(batch, M, P).to(A.device)

for b in range(batch):
    for m in range(M):
        sum = 0.0
        max_val = -torch.inf
        for p in range(P):
            for n in range(N):
                C[b, m, p] += A[b, m, n] * B[b, n, p]
            max_val = max(max_val, C[b, m, p])
        e = torch.exp(C[b, m] - max_val)
        C[b, m] = e / torch.sum(e, dim=-1)

return C.squeeze(0) if reduce else C

```

### Testing

There are several tests provided to ensure your operators are working.

To run our sample testing library, you can use the commands:

```bash
pytest tests/test_fused_ops.py
```

To test the speed/performance compared to unfused operators

```bash
python tests/test_fused_ops_perf.py
```
