import functools
from typing import Callable, Tuple, List

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
import math

import auto_diff as ad
import torch
from torchvision import datasets, transforms

max_len = 28

def transformer(X: ad.Node, nodes: List[ad.Node], 
                      model_dim: int, seq_length: int, input_dim: int, eps, batch_size, num_classes) -> ad.Node:
    """Construct the computational graph for a single transformer layer with sequence classification.

    Parameters
    ----------
    X: ad.Node
        A node in shape (batch_size, seq_length, input_dim), denoting the input data. seq_length = input_dim
    nodes: List[ad.Node]
        Nodes you would need to initialize the transformer.
    model_dim: int
        Dimension of the model (hidden size).
    seq_length: int
        Length of the input sequence.

    Returns
    -------
    output: ad.Node
        The output of the transformer layer, averaged over the sequence length for classification, in shape (batch_size, num_classes).
    """
    W_Q, W_K, W_V, W_O, W_1, b_1, W_2, b_2 = nodes

    # Compute Q, K, V -> (b, s, i) @  (b, i, m) = (b, s, m)
    Q = ad.matmul(X, ad.broadcast(W_Q, input_shape=[input_dim, model_dim], target_shape=[batch_size, input_dim, model_dim]))  # shape (batch, seq_len, model_dim)
    K = ad.matmul(X, ad.broadcast(W_K, input_shape=[input_dim, model_dim], target_shape=[batch_size, input_dim, model_dim]))  # shape (batch, seq_len, model_dim)
    V = ad.matmul(X, ad.broadcast(W_V, input_shape=[input_dim, model_dim], target_shape=[batch_size, input_dim, model_dim]))  # shape (batch, seq_len, model_dim)
    # Q*K^T / sqrt(d_k): (b, s, m) @ (b, m, s) -> (b, s, s)
    scores = ad.div_by_const(ad.matmul(Q, ad.transpose(K, -2, -1)), math.pow(model_dim, 0.5))
    # softmax over the last dimension and multiply attn_weights by V: (b, s, s) @ (b, s, m) = (b, s, m)
    attn_weights = ad.matmul(ad.softmax(scores - ad.max_op(scores, dim=-1, keepdim=True), dim=-1), V)
    # Attention output: (b, s, m) @ (b, m, m) = (b, s, m)
    attention_out = ad.matmul(attn_weights, ad.broadcast(W_O, input_shape=[model_dim, model_dim], target_shape=[batch_size, model_dim, model_dim]))

    # Layer Norm (b, s, m)
    X_out = ad.layernorm(attention_out, normalized_shape=[model_dim], eps=eps)

    # MLP 
    # (b, s, m) -> (b, s, m) -> (b, s, c)
    X_ = ad.matmul(X_out, ad.broadcast(W_1, input_shape=[model_dim, model_dim], target_shape=[batch_size, model_dim, model_dim])) + ad.broadcast(b_1, input_shape=[model_dim], target_shape=[batch_size, seq_length, model_dim])
    X_ = ad.relu(X_)
    X_ = ad.matmul(X_, ad.broadcast(W_2, input_shape=[model_dim, num_classes], target_shape=[batch_size, model_dim, num_classes])) + ad.broadcast(b_2, input_shape=[num_classes], target_shape=[batch_size, seq_length, num_classes])

    # Layer Norm (b, s, m)
    X_ = ad.layernorm(X_, normalized_shape=[num_classes], eps=eps)

    # Pooling (b, s, c) -> (b, c)
    logits = ad.mean(X_, dim=(1, ), keepdim=False)  # average across seq_length dimension

    return logits


def softmax_loss(Z: ad.Node, y_one_hot: ad.Node, batch_size: int) -> ad.Node:
    """Construct the computational graph of average softmax loss over
    a batch of logits.

    Parameters
    ----------
    Z: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        logits for the batch of instances.

    y_one_hot: ad.Node
        A node in of shape (batch_size, num_classes), containing the
        one-hot encoding of the ground truth label for the batch of instances.

    batch_size: int
        The size of the mini-batch.

    Returns
    -------
    loss: ad.Node
        Average softmax loss over the batch.
        When evaluating, it should be a zero-rank array (i.e., shape is `()`).

    Note
    ----
    1. In this homework, you do not have to implement a numerically
    stable version of softmax loss.
    2. You may find that in other machine learning frameworks, the
    softmax loss function usually does not take the batch size as input.
    Try to think about why our softmax loss may need the batch size.
    """
    # shape (b, c)
    log_prob = ad.log(ad.softmax(Z, dim=-1))
    # shape (b, c) -> (b, ) -> scalar
    cross_entropy_loss = ad.mean(-1.0 * ad.sum_op(ad.mul(y_one_hot, log_prob), dim=(-1,), keepdim=False), dim=(-1,), keepdim=False)
    return cross_entropy_loss

def clip_grad_norm(gradients, max_norm):
    """Clips gradients to have a maximum L2 norm.
    
    Parameters:
    - gradients: List[torch.Tensor], the raw gradient tensors.
    - max_norm: float, the maximum allowed norm.

    Returns:
    - List of clipped gradients.
    """
    total_norm = torch.sqrt(sum(torch.sum(g**2) for g in gradients))
    
    if total_norm > max_norm:
        scale_factor = max_norm / (total_norm + 1e-6)  # Avoid division by zero
        gradients = [g * scale_factor for g in gradients]
    
    return gradients

def sgd_epoch(
    f_run_model: Callable,
    X: torch.Tensor,
    y: torch.Tensor,
    model_weights: List[torch.Tensor],
    batch_size: int,
    lr: float,
) -> List[torch.Tensor]:
    """Run an epoch of SGD for the transformer encoder model
    on training data with regard to the given mini-batch size
    and learning rate.

    Parameters
    ----------
    f_run_model: Callable
        The function to run the forward and backward computation
        at the same time for transformer encoder model.
        It takes the training data, training label, model weight
        and bias as inputs, and returns the logits, loss value,
        weight gradient and bias gradient in order.
        Please check `f_run_model` in the `train_model` function below.

    X: torch.Tensor
        The training data in shape (num_examples, in_features).

    y: torch.Tensor
        The training labels in shape (num_examples,).

    model_weights: List[torch.Tensor]
        The model weights in the model.

    batch_size: int
        The mini-batch size.

    lr: float
        The learning rate.

    Returns
    -------
    model_weights: List[torch.Tensor]
        The model weights after update in this epoch.

    b_updated: torch.Tensor
        The model weight after update in this epoch.

    loss: torch.Tensor
        The average training loss of this epoch.
    """

    num_examples = X.shape[0]
    num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
    total_loss = 0.0

    for i in range(num_batches):
        # Get the mini-batch data
        start_idx = i * batch_size
        if start_idx + batch_size > num_examples:
            print('Skipped')
            continue
        end_idx = min(start_idx + batch_size, num_examples)
        X_batch = X[start_idx:end_idx, :max_len]
        y_batch = y[start_idx:end_idx]
           
        # Compute forward and backward passes
        logits, loss, *gradients = f_run_model([X_batch, y_batch] + model_weights)

        gradients = clip_grad_norm(gradients=gradients, max_norm=1.0)

        # In-place weight updates for efficiency
        for W, W_grad in zip(model_weights, gradients):
            W -= lr * W_grad  # In-place update
        
        # Accumulate the loss
        total_loss += loss.item()
    
    # u = [print(w.shape, w_grad.shape) for w, w_grad in zip(model_weights, gradients)]

    # Compute the average loss
    average_loss = total_loss / max(1, num_batches)
    print('Avg_loss:', average_loss)

    # You should return the list of parameters and the loss
    return model_weights, average_loss

def train_model():
    """Train a logistic regression model with handwritten digit dataset.

    Note
    ----
    Your implementation should NOT make changes to this function.
    """
    # Set up model params
    W_embed = ad.Variable("W_embed")
    W_Q = ad.Variable("W_Q")
    W_K = ad.Variable("W_K")
    W_V = ad.Variable("W_V")
    W_O = ad.Variable("W_O")
    W_1 = ad.Variable("W_1")
    b_1 = ad.Variable("b_1")
    W_2 = ad.Variable("W_2")
    b_2 = ad.Variable("b_2")
    model_params = [W_Q, W_K, W_V, W_O, W_1, b_1, W_2, b_2]


    # Tune your hyperparameters here
    # Hyperparameters
    input_dim = 28  # Each row of the MNIST image
    seq_length = max_len  # Number of rows in the MNIST image
    num_classes = 10 #
    model_dim = 128 #
    eps = 1e-05

    # - Set up the training settings.
    num_epochs = 20
    batch_size = 100
    lr = 0.07
    precision = torch.float64

    # Define the forward graph.
    X_input = ad.Variable('X')

    y_predict: ad.Node = transformer(X_input, nodes=model_params, model_dim=model_dim, seq_length=seq_length, input_dim=input_dim, eps=float(eps), batch_size=batch_size, num_classes=num_classes) # The output of the forward pass
    y_predict.name = "y_pred"

    y_groundtruth = ad.Variable(name="y")
    loss: ad.Node = softmax_loss(y_predict, y_groundtruth, batch_size)

    # Graph inputs
    training_inputs = [X_input, y_groundtruth] + model_params
    inference_inputs = [X_input] + model_params
    
    # Construct the backward graph.
    grads: List[ad.Node] = ad.gradients(loss, nodes=model_params)

    # Create the evaluator.
    evaluator = ad.Evaluator([y_predict, loss, *grads])
    test_evaluator = ad.Evaluator([y_predict])

    # - Load the dataset.
    #   Take 80% of data for training, and 20% for testing.
    # Prepare the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Convert the train dataset to NumPy arrays
    X_train = train_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_train = train_dataset.targets.numpy()

    # Convert the test dataset to NumPy arrays
    X_test = test_dataset.data.numpy().reshape(-1, 28 , 28) / 255.0  # Flatten to 784 features
    y_test = test_dataset.targets.numpy()

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False)  # Use sparse=False to get a dense array

    # Fit and transform y_train, and transform y_test
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))

    num_classes = 10

    # Initialize model weights.
    np.random.seed(0)
    stdv = 1.0 / np.sqrt(num_classes)
    W_Q_val = torch.from_numpy(np.random.uniform(-stdv, stdv, (input_dim, model_dim))).to(precision)
    W_K_val = torch.from_numpy(np.random.uniform(-stdv, stdv, (input_dim, model_dim))).to(precision)
    W_V_val = torch.from_numpy(np.random.uniform(-stdv, stdv, (input_dim, model_dim))).to(precision)
    W_O_val = torch.from_numpy(np.random.uniform(-stdv, stdv, (model_dim, model_dim))).to(precision)
    W_1_val = torch.from_numpy(np.random.uniform(-stdv, stdv, (model_dim, model_dim))).to(precision)
    W_2_val = torch.from_numpy(np.random.uniform(-stdv, stdv, (model_dim, num_classes))).to(precision)
    b_1_val = torch.from_numpy(np.random.uniform(-stdv, stdv, (model_dim,))).to(precision)
    b_2_val = torch.from_numpy(np.random.uniform(-stdv, stdv, (num_classes,))).to(precision)

    def f_run_model(model_weights):
        """The function to compute the forward and backward graph.
        It returns the logits, loss, and gradients for model weights.
        """
        result = evaluator.run(
            input_values={
                node: inp for node, inp in zip(training_inputs, model_weights)
            }
        )
        return result

    def f_eval_model(X_val, model_weights: List[torch.Tensor]):
        """The function to compute the forward graph only and returns the prediction."""
        num_examples = X_val.shape[0]
        num_batches = (num_examples + batch_size - 1) // batch_size  # Compute the number of batches
        # total_loss = 0.0
        all_logits = []
        for i in range(num_batches):
            # Get the mini-batch data
            start_idx = i * batch_size
            if start_idx + batch_size> num_examples:continue
            end_idx = min(start_idx + batch_size, num_examples)
            X_batch = X_val[start_idx:end_idx, :max_len]
            logits = test_evaluator.run(input_values={
                node: inp for node, inp in zip(inference_inputs, [X_batch] + model_weights)
            })
            all_logits.append(logits[0])
        # Concatenate all logits and return the predicted classes
        concatenated_logits = np.concatenate(all_logits, axis=0)
        predictions = np.argmax(concatenated_logits, axis=1)
        return predictions

    # Train the model.
    X_train, X_test, y_train, y_test= torch.tensor(X_train), torch.tensor(X_test), torch.DoubleTensor(y_train), torch.DoubleTensor(y_test)
    model_weights: List[torch.Tensor] = [W_Q_val, W_K_val, W_V_val, W_O_val, W_1_val, b_1_val, W_2_val, b_2_val] # Initialize the model weights here

    X_train = X_train.reshape(-1, seq_length, input_dim)
    X_test = X_test.reshape(-1, seq_length, input_dim)

    for epoch in range(num_epochs):
        X_train, y_train = shuffle(X_train.to(precision), y_train.to(precision))
        model_weights, loss_val = sgd_epoch(
            f_run_model, X_train, y_train, model_weights, batch_size, lr
        )

        # Evaluate the model on the test data.
        predict_label = f_eval_model(X_test.to(precision), model_weights)

        print(
            f"Epoch {epoch}: test accuracy = {np.mean(predict_label== y_test.numpy())}, "
            f"loss = {loss_val}"
        )

    # Return the final test accuracy.
    predict_label = f_eval_model(X_test.to(precision), model_weights)
    return np.mean(predict_label == y_test.numpy())


if __name__ == "__main__":
    print(f"Final test accuracy: {train_model()}")
