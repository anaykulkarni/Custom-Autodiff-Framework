from typing import List

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import auto_diff as ad

def check_compute_output(
    node: ad.Node, input_values: List[torch.Tensor], expected_output: torch.Tensor
) -> None:
    output = node.op.compute(node, input_values)
    torch.testing.assert_close(output, expected_output, atol=1e-4, rtol=1e-4)


def test_mul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.mul(x1, x2)

    check_compute_output(
        y,
        [
            torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            torch.tensor([[2.8, 0.7, -0.1, 0.0], [0.6, 6.6, 3.2, 3.1]]),
        ],
        torch.tensor([[-2.80, 1.40, -0.05, 0.00], [0.18, 0.00, -18.56, 9.61]]),
    )


def test_mul_by_const():
    x1 = ad.Variable("x1")
    y = ad.mul_by_const(x1, 2.7)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([[-2.70, 5.40, 1.35, 9.18], [0.81, 0.00, -15.66, 8.37]]),
    )


def test_div():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.div(x1, x2)

    check_compute_output(
        y,
        [
            torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]]),
            torch.tensor([[2.5, 4.0, -0.1, 0.1], [-8.0, 5.0, -2.5, -1.0]]),
        ],
        torch.tensor([[-0.4, 0.5, -5.0, 34.0], [-0.0375, 0.0, 2.32, -3.1]]),
    )


def test_div_by_const():
    x1 = ad.Variable("x1")
    y = ad.div_by_const(x1, 5.0)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.5, 3.4], [0.3, 0.0, -5.8, 3.1]])],
        torch.tensor([[-0.2, 0.4, 0.1, 0.68], [0.06, 0.0, -1.16, 0.62]]),
    )


def test_matmul():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.matmul(x1, x2)

    x1_val = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x2_val = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    check_compute_output(
        y,
        [x1_val, x2_val],
        torch.tensor([[27.0, 30.0, 33.0], [61.0, 68.0, 75.0], [95.0, 106.0, 117.0]]),
    )

def test_matmul_3d():
    x1 = ad.Variable("x1")
    x2 = ad.Variable("x2")
    y = ad.matmul(x1, x2)

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

    expected = torch.tensor([[[30.0, 36.0, 42.0],
                            [66.0, 81.0, 96.0],
                            [102.0, 126.0, 150.0]],
                           [[150.0, 126.0, 102.0],
                            [96.0, 81.0, 66.0],
                            [42.0, 36.0, 30.0]]])

    check_compute_output(
        y,
        [x1_val, x2_val],
        expected
    )


def test_layernorm():
    x = ad.Variable("x")
    y = ad.layernorm(x, normalized_shape=[3])

    check_compute_output(
        y,
        [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)],
        torch.tensor([[-1.224745, 0.0, 1.224745], [-1.224745, 0.0, 1.224745]], dtype=torch.float32)
    )


def test_relu():
    x = ad.Variable("x")
    y = ad.relu(x)

    check_compute_output(
        y,
        [torch.tensor([[-1.0, 2.0, 0.0], [3.0, -4.0, 5.0]], dtype=torch.float32)],
        torch.tensor([[0.0, 2.0, 0.0], [3.0, 0.0, 5.0]], dtype=torch.float32)
    )

def test_transpose():
    x = ad.Variable("x")
    y = ad.transpose(x, 1, 0)

    check_compute_output(
        y,
        [torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])],
        torch.tensor([[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]])
    )

def test_softmax():
    x = ad.Variable("x")
    y = ad.softmax(x)

    check_compute_output(
        y,
        [torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)],
        torch.tensor([[0.0900, 0.2447, 0.6652], [0.0900, 0.2447, 0.6652]], dtype=torch.float32)
    )

def test_broadcast():
    x = ad.Variable("x")
    y = ad.broadcast(x, input_shape=[3, 2], target_shape=[2, 3, 2])

    check_compute_output(
        y,
        [torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])],
        torch.tensor([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        ])
    )

def test_greater():
    x = ad.Variable("x")
    y = ad.Variable("y")
    z = ad.greater(x, y)

    x_val = torch.tensor([[-0.7494, -0.6661, -0.9543],
         [-0.4419,  1.9294,  0.3191]])
    y_val = torch.tensor([[-0.2219, -0.9706, -1.0460],
         [ 1.3733, -1.2985, -1.2688]])
    z_val = torch.tensor([[0., 1., 1.],
        [0., 1., 1.]])

    check_compute_output(
        z,
        [x_val, y_val],
        z_val
    )

def test_equal():
    x = ad.Variable("x")
    y = ad.Variable("y")
    z = ad.eq_op(x, y)

    x_val = torch.tensor([[-0.7494, -0.6661, -0.9543],
         [ 1.3733,  1.9294,  0.3191]])
    y_val = torch.tensor([[-0.2219, -0.6661, -1.0460],
         [ 1.3733, -1.2985, -1.2688]])
    z_val = torch.tensor([[0., 1., 0.],
        [1., 0., 0.]])

    check_compute_output(
        z,
        [x_val, y_val],
        z_val
    )

def test_max():
    x = ad.Variable("x")
    y = ad.max_op(x, dim=-1, keepdim=False)

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
    
    y_val = torch.tensor([[ 0.4141,  1.3651],
        [ 0.1472,  0.4971],
        [ 0.4794, -0.1622],
        [ 1.1805, -0.0787],
        [ 2.1054,  1.1774]])

    check_compute_output(
        y,
        [x_val],
        y_val
    )

def test_max2():
    x = ad.Variable("x")
    y = ad.max_op(x, dim=-1, keepdim=True)

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
    
    y_val = torch.tensor([[[ 0.4141],
         [ 1.3651]],

        [[ 0.1472],
         [ 0.4971]],

        [[ 0.4794],
         [-0.1622]],

        [[ 1.1805],
         [-0.0787]],

        [[ 2.1054],
         [ 1.1774]]])

    check_compute_output(
        y,
        [x_val],
        y_val
    )


if __name__ == "__main__":
    test_mul()
    test_mul_by_const()
    test_div()
    test_div_by_const()
    test_layernorm()
    test_relu()
    test_softmax()
    test_matmul()
    test_matmul_3d()
    test_transpose()
    test_broadcast()
    test_greater()
    test_equal()
    test_max()
    test_max2()