from typing import Dict, List

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import auto_diff as ad

def check_evaluator_output(
    evaluator: ad.Evaluator,
    input_values: Dict[ad.Node, torch.Tensor],
    expected_outputs: List[torch.Tensor],
) -> None:
    output_values = evaluator.run(input_values)
    assert len(output_values) == len(expected_outputs)
    for output_val, expected_val in zip(output_values, expected_outputs):
        print(repr(output_val))
        torch.testing.assert_close(output_val, expected_val, atol=1e-4, rtol=1e-4)


def test_mul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.mul(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
            y_grad: torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
        },
        expected_outputs=[
            torch.tensor([[-1.12, 0.35, 0.5, 0.0], [-0.0225, 0.0, 7.424, -9.61]]),
            torch.tensor([[0.4, 1.0, -2.5, 115.6], [-0.01125, 0.0, -13.456, -9.61]]),
        ],
    )


def test_div():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.div(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            x2: torch.tensor([[2.5, 4.0, -0.1, 0.1], [-8.0, 5.0, -2.5, -1.0]]),
            y_grad: torch.ones((2, 4), dtype=torch.float32),
        },
        expected_outputs=[
            torch.tensor([[0.4, 0.25, -10.0, 10.0], [-0.125, 0.2, -0.4, -1.0]]),
            torch.tensor([[0.16, -0.125, -50.0, -340.0], [-0.0046875, 0, 0.928, -3.1]]),
        ],
    )


def test_div_by_const():
    x1 = ad.Variable("x1")
    y = ad.div_by_const(x1, 5.0)
    evaluator = ad.Evaluator(eval_nodes=[y])

    check_evaluator_output(
        evaluator,
        input_values={x1: torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])},
        expected_outputs=[torch.tensor([[-0.2, 0.4, 0.1, 0.68], [0.06, 0.0, -1.16, 0.62]])],
    )

def test_matmul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.matmul(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    x1_val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x2_val = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    y_grad_val = torch.ones((3, 3), dtype=torch.float32)
    x1_grad_expected = torch.tensor([[24.0, 33.0], [24.0, 33.0], [24.0, 33.0]])
    x2_grad_expected = torch.tensor([[9.0, 9.0, 9.0], [12.0, 12.0, 12.0]])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: x1_val,
            x2: x2_val,
            y_grad: y_grad_val,
        },
        expected_outputs=[x1_grad_expected, x2_grad_expected],
    )

def test_matmul_3d():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.matmul(x1, x2)
    y_grad = ad.Variable("y_grad")
    x1_grad, x2_grad = y.op.gradient(y, y_grad)
    evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

    x1_val = torch.tensor([[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]],
                          [[9.0, 8.0, 7.0],
                           [6.0, 5.0, 4.0],
                           [3.0, 2.0, 1.0]]])
    
    x2_val = torch.tensor([[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0],
                           [7.0, 8.0, 9.0]],
                          [[9.0, 8.0, 7.0],
                           [6.0, 5.0, 4.0],
                           [3.0, 2.0, 1.0]]])

    y_grad_val = torch.ones((2, 3, 3), dtype=torch.float32)

    x1_grad_expected = torch.tensor([[[6.0, 15.0, 24.0],
                                    [6.0, 15.0, 24.0],
                                    [6.0, 15.0, 24.0]],
                                   [[24.0, 15.0, 6.0],
                                    [24.0, 15.0, 6.0],
                                    [24.0, 15.0, 6.0]]])

    x2_grad_expected = torch.tensor([[[12.0, 12.0, 12.0],
                                    [15.0, 15.0, 15.0],
                                    [18.0, 18.0, 18.0]],
                                   [[18.0, 18.0, 18.0],
                                    [15.0, 15.0, 15.0],
                                    [12.0, 12.0, 12.0]]])

    check_evaluator_output(
        evaluator,
        input_values={
            x1: x1_val,
            x2: x2_val,
            y_grad: y_grad_val,
        },
        expected_outputs=[x1_grad_expected, x2_grad_expected],
    )

# def test_matmul_2d3d():
#     x1 = ad.Variable("x1")
#     x2 = ad.Variable("x2")
#     y = ad.matmul(x1, x2)
#     y_grad = ad.Variable("y_grad")
#     x1_grad, x2_grad = y.op.gradient(y, y_grad)
#     evaluator = ad.Evaluator(eval_nodes=[x1_grad, x2_grad])

#     x1_val = torch.tensor(
#         [[[-0.8010,  0.8222, -1.6132, -0.2925],
#          [ 0.0911,  1.6612, -0.0201,  1.0275],
#          [-0.2434,  2.3695,  1.0944, -1.8693]],

#         [[-1.0510,  1.1515,  0.8505,  1.1467],
#          [ 0.7488,  0.2565,  0.8228, -1.6832],
#          [ 0.7987, -0.4858,  0.1787, -0.4922]]], dtype=torch.float32)
    
#     x2_val = torch.tensor(
#         [[ 0.1702, -0.6873, -1.7630],
#         [-0.7935,  0.6920,  0.9275],
#         [-0.8229,  1.4516, -0.5341],
#         [ 0.2990, -0.6090, -0.4993]], dtype=torch.float32)
    
#     y_grad_val = torch.ones(2, 3, 3)

#     x1_grad_expected = torch.tensor(
#         [[[-2.2801,  0.8260,  0.0947, -0.8094],
#          [-2.2801,  0.8260,  0.0947, -0.8094],
#          [-2.2801,  0.8260,  0.0947, -0.8094]],

#         [[-2.2801,  0.8260,  0.0947, -0.8094],
#          [-2.2801,  0.8260,  0.0947, -0.8094],
#          [-2.2801,  0.8260,  0.0947, -0.8094]]], dtype=torch.float32)
    
#     x2_grad_expected = torch.tensor(
#         [[-0.4569, -0.4569, -0.4569],
#         [ 5.7751,  5.7751,  5.7751],
#         [ 1.3131,  1.3131,  1.3131],
#         [-2.1631, -2.1631, -2.1631]], dtype=torch.float32)

#     check_evaluator_output(
#         evaluator,
#         input_values={
#             x1: x1_val,
#             x2: x2_val,
#             y_grad: y_grad_val,
#         },
#         expected_outputs=[x1_grad_expected, x2_grad_expected],
#     )


def test_layernorm():
    x = ad.Variable("x")
    y = ad.layernorm(x, normalized_shape=[3])
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    y_grad_val = torch.tensor([[12, 4, 2], [-3, -5, 3]], dtype=torch.float32)

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[
            torch.tensor([
                [1.2248, -2.4495,  1.2246],
                [2.0412, -4.0825, 2.0413]
            ], dtype=torch.float32)
        ]
    )

def test_relu():
    x = ad.Variable("x")
    y = ad.relu(x)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[-1.0, 2.0, 0.0], [3.0, -4.0, 5.0]], dtype=torch.float32)
    y_grad_val = torch.ones_like(x_val)

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=torch.float32)]
    )

def test_softmax():
    x = ad.Variable("x")
    y = ad.softmax(x)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    y_grad_val = torch.tensor([[0.5, -0.3, 0.8], [-0.2, 0.4, -0.1]], dtype=torch.float32)

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[
            torch.tensor([
                [-0.0003, -0.1967,  0.1971],
                [-0.0192,  0.0946, -0.0754]
            ], dtype=torch.float32)
        ]
    )

def test_transpose():
    x = ad.Variable("x")
    y = ad.transpose(x, 1, 0)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_grad_val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])]
    )

def test_broadcast():
    x = ad.Variable("x")
    y = ad.broadcast(x, input_shape=[3, 2], target_shape=[2, 3, 2])
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])

    x_val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_grad_val = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], 
        [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
    ])

    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[torch.tensor([[8.0, 10.0], [12.0, 14.0], [16.0, 18.0]])]
    )

# ------------------------------------------------------------------------------
# 1. Basic test: 2D input with normalized_shape=[3]
# ------------------------------------------------------------------------------

def test_layernorm_basic():
    # Create variables and forward pass with layernorm over the last dimension (3)
    x = ad.Variable("x")
    y = ad.layernorm(x, normalized_shape=[3])
    y_grad = ad.Variable("y_grad")
    
    # Compute gradient with respect to x
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])
    
    # Define a 2x3 input and corresponding upstream gradient
    x_val = torch.tensor([[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]], dtype=torch.float32)
    y_grad_val = torch.tensor([[12.0, 4.0, 2.0],
                               [-3.0, -5.0, 3.0]], dtype=torch.float32)
    
    # Expected gradient computed manually (see derivation in notes)
    expected_x_grad = torch.tensor([
        [1.2248, -2.4495, 1.2246],
        [2.0412, -4.0825, 2.0413]
    ], dtype=torch.float32)
    
    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[expected_x_grad]
    )


# ------------------------------------------------------------------------------
# 2. 3D input test: Apply layernorm on each 1D group in the last dimension
# ------------------------------------------------------------------------------

def test_layernorm_3d():
    # Here x has shape [2, 2, 3]. With normalized_shape=[3], the normalization
    # is applied over each 1D vector (last dim) independently.
    x = ad.Variable("x")
    y = ad.layernorm(x, normalized_shape=[3])
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])
    
    # Define a 3D input and upstream gradient:
    x_val = torch.tensor([
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]],
        [[-1.0,  0.0,  1.0],
         [ 2.0,  3.0,  4.0]]
    ], dtype=torch.float32)
    
    y_grad_val = torch.tensor([
        [[12.0, 4.0, 2.0],
         [1.0, 3.0, 5.0]],
        [[-3.0, -5.0, 3.0],
         [7.0, 8.0, 9.0]]
    ], dtype=torch.float32)
    
    # For each 1D group the gradient is computed as:
    #    dx = (dy - mean(dy) - (x-mu)*mean(dy*(x-mu))/(var+eps)) / sqrt(var+eps)
    # In our chosen numbers, the second group in each sample turns out to have
    # upstream gradients that cancel (i.e. expected gradient is 0).
    expected_x_grad = torch.tensor([
        [[1.2247, -2.4495, 1.2247],
         [0.0, 0.0, 0.0]],
        [[2.0412, -4.0825, 2.0412],
         [0.0, 0.0, 0.0]]
    ], dtype=torch.float32)
    
    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[expected_x_grad]
    )


# ------------------------------------------------------------------------------
# 3. Constant-input test: When all x-values are equal the variance is zero,
#    so the eps term dominates in the normalization denominator.
# ------------------------------------------------------------------------------

def test_layernorm_constant():
    # For a constant input, mu = constant and (x - mu) = 0 so the forward
    # output becomes 0. The gradient simplifies to:
    #    dx = (dy - mean(dy)) / sqrt(eps)
    # (assuming the default eps=1e-5, so sqrt(eps) ~ 0.0031623)
    x = ad.Variable("x")
    y = ad.layernorm(x, normalized_shape=[3])
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])
    
    # Constant input (1x3) and an arbitrary upstream gradient.
    x_val = torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float32)
    y_grad_val = torch.tensor([[2.0, -1.0, 3.0]], dtype=torch.float32)
    
    # Compute mean of y_grad for the group: (2 - 1 + 3) / 3 = 4/3 ≈ 1.33333.
    # Then expected gradient: (y_grad - 1.33333) / sqrt(1e-5)
    # That is approximately: [(0.66667/0.0031623), (-2.33333/0.0031623), (1.66667/0.0031623)]
    # ≈ [211.0, -737.0, 527.0]
    expected_x_grad = torch.tensor([[210.818075, -737.858521, 527.043608]], dtype=torch.float32)
    
    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[expected_x_grad]
    )


# ------------------------------------------------------------------------------
# 4. Full-sample normalization: normalized_shape spans multiple dimensions.
#    Here the normalization is applied over the entire per-sample tensor.
# ------------------------------------------------------------------------------

def test_layernorm_full_normalization():
    # In this test the input x has shape [2,2,3] and we set
    # normalized_shape=[2, 3]. That means for each sample (the first dim)
    # the entire [2,3] block is normalized.
    x = ad.Variable("x")
    y = ad.layernorm(x, normalized_shape=[2, 3])
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])
    
    x_val = torch.tensor([
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]],
        [[7.0, 8.0, 9.0],
         [10.0, 11.0, 12.0]]
    ], dtype=torch.float32)
    
    y_grad_val = torch.tensor([
        [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]],
        [[-1.0, -2.0, -3.0],
         [-4.0, -5.0, -6.0]]
    ], dtype=torch.float32)
    
    # For the first sample, note that:
    #   - The flattened x[0] is [1,2,3,4,5,6] with mean 3.5 and sigma ≈ 1.7078.
    #   - The upstream gradient y_grad[0] also sums to 21 so its mean is 3.5.
    # It turns out that (y_grad - mean - (x-mu)*(...)) simplifies to (y_grad - x),
    # and in our case (y_grad - x) is zero for the first sample.
    # A similar cancellation happens for the second sample.
    expected_x_grad = torch.zeros_like(x_val)
    
    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[expected_x_grad]
    )

# ------------------------------------------------------------------------------
# 5. Test Square root op
# ------------------------------------------------------------------------------
def test_sqrt():
    x = ad.Variable("x")
    y = ad.sqrt(x)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])
    
    x_val = torch.tensor([[0.4132, 0.4985, 0.9466],
        [0.0209, 0.0256, 0.1550]], dtype=torch.float32)
    
    y_grad_val = torch.tensor([[-0.6845, -1.7382, -0.0224],
        [-0.0380, -0.4481, -0.0794]], dtype=torch.float32)
    
    expected_x_grad = torch.tensor([[-0.5324, -1.2309, -0.0115],
        [-0.1314, -1.4003, -0.1008]], dtype=torch.float32)
    
    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[expected_x_grad]
    )

# ------------------------------------------------------------------------------
# 6. Test Mean Op: keep dim False
# ------------------------------------------------------------------------------
def test_mean1():
    x = ad.Variable("x")
    y = ad.mean(x, dim=(1, 2), keepdim=False)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])
    
    x_val = torch.tensor([[[-0.1143,  0.5332,  0.4219, -0.8115],
          [-0.3144,  0.5548,  0.9968,  1.8047],
          [-0.2028, -1.2034, -1.0559,  0.5434]],
 
         [[-0.2839, -0.7814,  0.7064, -0.5365],
          [ 0.4399,  0.1988,  0.1510,  1.7313],
          [ 1.2421, -1.9284, -0.7935,  0.7951]]], dtype=torch.float32)
    
    y_grad_val = torch.tensor([0.2012, 0.0855], dtype=torch.float32)
    
    expected_x_grad = torch.tensor([[[0.0168, 0.0168, 0.0168, 0.0168],
          [0.0168, 0.0168, 0.0168, 0.0168],
          [0.0168, 0.0168, 0.0168, 0.0168]],
 
         [[0.0071, 0.0071, 0.0071, 0.0071],
          [0.0071, 0.0071, 0.0071, 0.0071],
          [0.0071, 0.0071, 0.0071, 0.0071]]], dtype=torch.float32)
    
    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[expected_x_grad]
    )

# ------------------------------------------------------------------------------
# 7. Test Mean Op: keep dim True
# ------------------------------------------------------------------------------
def test_mean2():
    x = ad.Variable("x")
    y = ad.mean(x, dim=(1, 2), keepdim=True)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])
    
    x_val = torch.tensor([[[ 0.7114,  2.1445, -0.9114, -0.7149],
          [-0.3072,  1.7361,  0.2153, -0.5989],
          [-0.2515, -0.8363,  0.9622,  1.6980]],
 
         [[ 0.6066, -0.3146,  1.2215,  0.8129],
          [ 0.5621, -0.5687,  0.1967, -0.3326],
          [ 0.0896, -1.9020,  1.2880, -0.6730]]], dtype=torch.float32)
    
    y_grad_val = torch.tensor([[[0.0644]],[[0.7459]]], dtype=torch.float32)
    
    expected_x_grad = torch.tensor([[[0.0054, 0.0054, 0.0054, 0.0054],
          [0.0054, 0.0054, 0.0054, 0.0054],
          [0.0054, 0.0054, 0.0054, 0.0054]],
 
         [[0.0622, 0.0622, 0.0622, 0.0622],
          [0.0622, 0.0622, 0.0622, 0.0622],
          [0.0622, 0.0622, 0.0622, 0.0622]]], dtype=torch.float32)
    
    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[expected_x_grad]
    )


# ------------------------------------------------------------------------------
# 6. Test Variance Op: keep dim False
# ------------------------------------------------------------------------------
def test_var1():
    x = ad.Variable("x")
    y = ad.var_op(x, dim=(1, 2), keepdim=False)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])
    
    x_val = torch.tensor([[[-0.2101,  0.1864, -0.4722,  0.4198],
          [ 0.6548,  0.3948,  0.3682,  1.0939],
          [ 0.4836, -0.4942, -0.5057, -0.1759]],
 
         [[ 0.0882, -0.9525,  0.5763,  2.0846],
          [-0.6341, -0.5233, -0.2569,  3.4167],
          [ 0.5320,  0.5112, -0.2976, -1.2177]]], dtype=torch.float32)
    
    y_grad_val = torch.tensor([0.2034, 0.5103], dtype=torch.float32)
    
    expected_x_grad = torch.tensor([[[-0.0120,  0.0014, -0.0209,  0.0093],
          [ 0.0173,  0.0085,  0.0076,  0.0321],
          [ 0.0115, -0.0217, -0.0221, -0.0109]],
 
         [[-0.0161, -0.1046,  0.0254,  0.1537],
          [-0.0775, -0.0681, -0.0454,  0.2670],
          [ 0.0217,  0.0199, -0.0489, -0.1271]]], dtype=torch.float32)
    
    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[expected_x_grad]
    )

# ------------------------------------------------------------------------------
# 7. Test Variance Op: keep dim True
# ------------------------------------------------------------------------------
def test_var2():
    x = ad.Variable("x")
    y = ad.var_op(x, dim=(1, 2), keepdim=True)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])
    
    x_val = torch.tensor([[[ 0.2176, -0.1787,  0.6116,  0.2742],
          [ 0.0589,  0.7558,  0.6080,  0.5808],
          [-0.1984, -1.0451,  0.7399,  0.6280]],
 
         [[-1.3336,  0.8790, -1.1031, -0.8845],
          [ 0.0076,  1.5554, -1.4078, -0.5486],
          [ 1.0254, -0.6128, -0.2645,  0.0631]]], dtype=torch.float32)
    
    y_grad_val = torch.tensor([[[0.4733]],
         [[0.5970]]], dtype=torch.float32)
    
    expected_x_grad = torch.tensor([[[-0.0029, -0.0342,  0.0282,  0.0016],
          [-0.0154,  0.0396,  0.0279,  0.0257],
          [-0.0357, -0.1025,  0.0383,  0.0295]],
 
         [[-0.1109,  0.1092, -0.0880, -0.0662],
          [ 0.0225,  0.1765, -0.1183, -0.0328],
          [ 0.1238, -0.0392, -0.0046,  0.0280]]], dtype=torch.float32)
    
    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[expected_x_grad]
    )

# ------------------------------------------------------------------------------
# 8. Test Sum Op: keep dim False
# ------------------------------------------------------------------------------
def test_sumop1():
    x = ad.Variable("x")
    y = ad.sum_op(x, dim=(1, 2), keepdim=False)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])
    
    x_val = torch.tensor([[[ 0.5700, -1.2243, -0.0556,  0.2402],
          [ 1.0615,  1.0085,  0.1583, -0.1945],
          [ 1.2226, -0.1799, -1.3312,  0.7779]],
 
         [[-2.5826, -0.3517,  1.9603, -0.4139],
          [-1.0944, -0.7608, -0.2329, -1.6023],
          [-0.0353, -0.1969, -0.9676,  2.6742]]], dtype=torch.float32)
    
    y_grad_val = torch.tensor([0.8600, 0.0319], dtype=torch.float32)
    
    expected_x_grad = torch.tensor([[[0.8600, 0.8600, 0.8600, 0.8600],
          [0.8600, 0.8600, 0.8600, 0.8600],
          [0.8600, 0.8600, 0.8600, 0.8600]],
 
         [[0.0319, 0.0319, 0.0319, 0.0319],
          [0.0319, 0.0319, 0.0319, 0.0319],
          [0.0319, 0.0319, 0.0319, 0.0319]]], dtype=torch.float32)
    
    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[expected_x_grad]
    )

# ------------------------------------------------------------------------------
# 9. Test Sum Op: keep dim True
# ------------------------------------------------------------------------------
def test_sumop2():
    x = ad.Variable("x")
    y = ad.sum_op(x, dim=(1, 2), keepdim=True)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])
    
    x_val = torch.tensor([[[ 0.2628, -1.2706,  0.5243,  1.0938],
          [ 0.1224, -0.4609,  0.1006,  0.6241],
          [ 0.8196, -0.7017,  0.9935,  0.5064]],
 
         [[-0.4242,  2.0392,  1.1600, -0.1190],
          [ 2.0171, -0.7825,  0.9106,  1.0935],
          [-1.0033,  0.7160,  0.8587, -0.7994]]], dtype=torch.float32)
    
    y_grad_val = torch.tensor([[[0.5281]],
         [[0.2617]]], dtype=torch.float32)
    
    expected_x_grad = torch.tensor([[[0.5281, 0.5281, 0.5281, 0.5281],
          [0.5281, 0.5281, 0.5281, 0.5281],
          [0.5281, 0.5281, 0.5281, 0.5281]],
 
         [[0.2617, 0.2617, 0.2617, 0.2617],
          [0.2617, 0.2617, 0.2617, 0.2617],
          [0.2617, 0.2617, 0.2617, 0.2617]]], dtype=torch.float32)
    
    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[expected_x_grad]
    )   

def test_maxop1():
    x = ad.Variable("x")
    y = ad.max_op(x, dim=-1, keepdim=False)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])
    
    x_val = torch.tensor([[[-2.1660, -0.4407,  0.4141],
         [ 1.3651,  0.8741,  1.2153]],

        [[-0.5492,  0.1472, -0.8440],
         [-1.4951,  0.4971,  0.0824]],

        [[ 0.4794, -0.8752, -1.1831],
         [-0.1622, -0.1622, -0.8383]],

        [[-1.0575,  1.1805, -0.5331],
         [-0.0787, -2.4893, -0.4439]],

        [[ 2.1054,  1.2464, -0.9066],
         [ 1.0460,  1.1774,  0.7688]]])
    
    y_grad_val = torch.tensor([[ 0.5611,  0.8468],
        [ 1.1085, -0.0027],
        [-0.1348, -0.0764],
        [ 0.3667,  2.4105],
        [-1.3616,  2.3695]])
    
    expected_x_grad = torch.tensor(
        [[[ 0.0000,  0.0000,  0.5611],
         [ 0.8468,  0.0000,  0.0000]],

        [[ 0.0000,  1.1085,  0.0000],
         [ 0.0000, -0.0027,  0.0000]],

        [[-0.1348,  0.0000,  0.0000],
         [-0.0382,  -0.0382,  0.0000]],

        [[ 0.0000,  0.3667,  0.0000],
         [ 2.4105,  0.0000,  0.0000]],

        [[-1.3616,  0.0000,  0.0000],
         [ 0.0000,  2.3695,  0.0000]]], dtype=torch.float32)
    
    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[expected_x_grad]
    )

def test_maxop2():
    x = ad.Variable("x")
    y = ad.max_op(x, dim=-1, keepdim=True)
    y_grad = ad.Variable("y_grad")
    x_grad = y.op.gradient(y, y_grad)[0]
    evaluator = ad.Evaluator(eval_nodes=[x_grad])
    
    x_val = torch.tensor([[[-2.1660, -0.4407,  0.4141],
         [ 1.3651,  0.8741,  1.2153]],

        [[-0.5492,  0.1472, -0.8440],
         [-1.4951,  0.4971,  0.0824]],

        [[ 0.4794, -0.8752, -1.1831],
         [-0.1622, -0.1622, -0.8383]],

        [[-1.0575,  1.1805, -0.5331],
         [-0.0787, -2.4893, -0.4439]],

        [[ 2.1054,  1.2464, -0.9066],
         [ 1.0460,  1.1774,  0.7688]]])
    
    y_grad_val = torch.tensor([[[-0.3860],
         [-1.0542]],

        [[ 0.9653],
         [-0.6724]],

        [[-2.2496],
         [ 0.8408]],

        [[-0.3336],
         [-1.3273]],

        [[ 1.0801],
         [ 0.6309]]])
    
    expected_x_grad = torch.tensor([[[ 0.0000,  0.0000, -0.3860],
         [-1.0542,  0.0000,  0.0000]],

        [[ 0.0000,  0.9653,  0.0000],
         [ 0.0000, -0.6724,  0.0000]],

        [[-2.2496,  0.0000,  0.0000],
         [ 0.4204,  0.4204,  0.0000]],

        [[ 0.0000, -0.3336,  0.0000],
         [-1.3273,  0.0000,  0.0000]],

        [[ 1.0801,  0.0000,  0.0000],
         [ 0.0000,  0.6309,  0.0000]]])
    
    check_evaluator_output(
        evaluator,
        input_values={x: x_val, y_grad: y_grad_val},
        expected_outputs=[expected_x_grad]
    )

if __name__ == "__main__":
    test_mul()
    test_div()
    test_layernorm()
    test_relu() 
    test_softmax()
    test_matmul()
    test_transpose()
    test_broadcast()
    test_layernorm_basic()
    test_layernorm_3d()
    test_layernorm_constant()
    test_layernorm_full_normalization()
    test_sqrt()
    test_mean1()
    test_mean2()
    test_var1()
    test_var2()
    test_sumop1()
    test_sumop2()
    test_maxop1()
    test_maxop2()

