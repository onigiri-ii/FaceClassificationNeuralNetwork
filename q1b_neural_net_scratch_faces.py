"""Template for a 3 layer feed forward neural network for digit
classification, implemented from scratch.

Architecture: input -> hidden1 -> hidden2 -> output.

Implement the forward pass, back propagation, and weight update
yourself. You may use numpy for linear algebra. You may not use torch,
tensorflow, sklearn, jax, or keras for training, gradients, or
prediction.

Required public API (fixed for auto grading):
  * class `ScratchNeuralNetworkDigits` with methods `forward`,
    `backward`, `update_weights`, `train`, `predict`, `evaluate`.
  * `main(training_percent: int, num_iterations: int = 5)`.

Usage:
    python3 q1b_neural_net_scratch_digits.py <training_percent>
"""

import sys
import time
import numpy as np

from util_digits import load_digits, flatten_images


class ScratchNeuralNetworkDigits:
    """3 layer fully connected network: 784 to h1 to h2 to 10.

    Use any reasonable hidden activation (ReLU, sigmoid, tanh). For the
    output layer, softmax paired with cross entropy loss is typical.
    Document your choices in the report.

    Implementation notes:
      * Store weights as numpy arrays: W1 (784, h1), W2 (h1, h2),
        W3 (h2, 10), plus biases b1, b2, b3.
      * Initialise with small random values (scaled Gaussian, He,
        Xavier) to break symmetry.
      * `forward` should cache intermediate activations so that
        `backward` can compute gradients without re running forward.
    """

    def __init__(
        self,
        input_size: int = 28 * 28,
        hidden1_size: int = 128,
        hidden2_size: int = 64,
        output_size: int = 10,
        learning_rate: float = 0.05,
        num_epochs: int = 50,
        batch_size: int = 32,
        seed: int | None = None,
    ):
        """Initialise network hyperparameters and weight matrices."""
        self.input_size = input_size
        self.h1_size = hidden1_size
        self.h2_size = hidden2_size
        self.output_size = output_size
        self.lr = learning_rate
        self.epochs = num_epochs
        self.batches = batch_size
        self.seed = seed
        self.cache = None

        self.weights = {
            'W1': np.random.randn(hidden1_size, input_size) * np.sqrt(1. / input_size),
            'b1': np.zeros((self.h1_size, 1)),
            'W2': np.random.randn(hidden2_size, hidden1_size) * np.sqrt(1. / hidden1_size),
            'b2': np.zeros((self.h2_size, 1)),
            'W3': np.random.randn(output_size, hidden2_size) * np.sqrt(1. / hidden2_size),
            'b3': np.zeros((output_size, 1)),
        }
    
    def relu(self, x) -> float:
        """ReLU activation function, used for fast non-linearization of hidden layers"""
        return np.maximum(0,x)
    
    def softmax(self, x) -> float:
        """Softmax activation function, used for output classification
        (multiple classes, digits 0-9)"""
        exp = np.exp(x - np.max(x, axis=0, keepdims=True))
        return exp / np.sum(exp, axis=0, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass.

        `X` has shape (N, 784). Return shape is (N, 10). You may return
        probabilities (after softmax) or raw logits; keep `predict` and
        `backward` consistent with your choice.
        """

        # Gets weights and biases for each layer
        W1, b1 = self.weights['W1'], self.weights['b1']
        W2, b2 = self.weights['W2'], self.weights['b2']
        W3, b3 = self.weights['W3'], self.weights['b3']

        # 1. Dot product of weight matrix and current layer
        # 2. Activation function applied, giving next layer
        z1 = np.dot(W1, X.T) + b1
        a1 = self.relu(z1)
        z2 = np.dot(W2, a1) + b2
        a2 = self.relu(z2)
        z3 = np.dot(W3, a2) + b3
        a3 = self.softmax(z3)

        # Keep for backpropagation
        self.cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
        return a3
    
    def loss(self, pred, act) -> float:
        """Calculate the loss between the expected and observed values.
        
        pred: contains expected values
        act: contains observed values from forward pass

        Since y is a 10-dim vector, this sum calculates all 10 cases simultaneously.
        Adding 1e-12 prevents the log from being 0.
        """
        loss = -1/(len(pred[0])) * np.sum(pred * np.log(act + 1e-12))
        return loss

    def backward(self, X: np.ndarray, y_onehot: np.ndarray) -> dict:
        """Back propagate loss gradients through the network.

        `X` has shape (N, 784); `y_onehot` has shape (N, 10). Return a
        dict like
        `{"dW1": ..., "db1": ..., "dW2": ..., "db2": ..., "dW3": ..., "db3": ...}`.
        """
        (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = self.cache
        d = len(X)

        # Calculate derivatives
        dz3 = a3 - y_onehot
        dW3 = 1/d * np.dot(dz3, a2.T)
        db3 = 1/d * np.sum(dz3, axis=1, keepdims=True)

        da2 = np.dot(W3.T, dz3)
        dz2 = da2 * (a2 > 0)
        dW2 = 1/d * np.dot(dz2, a1.T)
        db2 = 1/d * np.sum(dz2, axis=1, keepdims=True)

        da1 = np.dot(W2.T, dz2)
        dz1 = da1 * (a1 > 0)
        dW1 = 1/d * np.dot(dz1, X)
        db1 = 1/d * np.sum(dz1, axis=1, keepdims=True)

        # Keep gradients
        grads = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2,
            'dW3': dW3,
            'db3': db3
        }

        return grads

    def update_weights(self, grads: dict) -> None:
        """Apply one gradient descent step using `grads` from `backward`."""
        self.weights['W1'] -= self.lr * grads['dW1']
        self.weights['b1'] -= self.lr * grads['db1']
        self.weights['W2'] -= self.lr * grads['dW2']
        self.weights['b2'] -= self.lr * grads['db2']
        self.weights['W3'] -= self.lr * grads['dW3']
        self.weights['b3'] -= self.lr * grads['db3']

    def train(self, training_images: np.ndarray, training_labels: np.ndarray) -> None:
        """Full training loop: epochs and mini batches.

        `training_images` has shape (N, 28, 28). `training_labels` has
        shape (N,) with values in {0..9}.
        """
        # Labels get flattened into matrix of size 10 (for each category)
        # Each submatrix is a label for the corresponding category
        training_labels_flat = np.zeros((10, training_labels.size))
        training_labels_flat[training_labels, np.arange(training_labels.size)] = 1
        
        # Train appropriate number of epochs
        for epoch in range(self.epochs):

            indices = np.random.permutation(len(training_images))
            X_shuffled = training_images[indices]
            y_shuffled = training_labels_flat[:, indices]

            for start in range(0, len(training_images), self.batches):

                end = start + self.batches

                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[:, start:end]

                output = self.forward(X_batch)
                loss = self.loss(y_batch, output)
                grads = self.backward(X_batch, y_batch)
                self.update_weights(grads)

    def predict(self, image: np.ndarray) -> int:
        """Predict a single label in {0..9} for a 28x28 image."""
        # No back-prop needed, weights don't get updated
        res = self.forward(image)

        # Convert model outputs to same format as input labels
        predictions = np.argmax(res, axis=0)

        return predictions[0]

    def evaluate(self, images: np.ndarray, labels: np.ndarray) -> float:
        """Return classification accuracy on a batch of images."""
        # No back-prop needed, weights don't get updated
        res = self.forward(images)

        # Convert model outputs to same format as input labels
        predictions = np.argmax(res, axis=0)
        
        # Calculate accuracy of model's predictions
        # Cross compares with actual labels for dataset
        sum = 0
        for i in range(len(predictions)):
            if (predictions[i] == labels[i]): sum += 1

        return sum / len(predictions)


def main(training_percent: int, num_iterations: int = 5) -> dict:
    """Run the standard train/test pipeline for the scratch NN on digits."""
    training_images, training_labels = load_digits("training")
    test_images, test_labels = load_digits("test")

    # Images get flattened for input in NN
    train_imgs_flat = flatten_images(training_images)
    test_imgs_flat = flatten_images(test_images)

    num_total = len(training_images)
    sample_size = (num_total * training_percent) // 100

    train_times = np.zeros(num_iterations)
    accuracies = np.zeros(num_iterations)

    for i in range(num_iterations):
        idx = np.random.choice(num_total, size=sample_size, replace=False)
        x_sample = train_imgs_flat[idx]
        y_sample = training_labels[idx]

        net = ScratchNeuralNetworkDigits()
        start = time.time()
        net.train(x_sample, y_sample)
        train_times[i] = time.time() - start

        accuracies[i] = net.evaluate(test_imgs_flat, test_labels)

    errors = 1.0 - accuracies
    results = {
        "training_percent": training_percent,
        "mean_train_time": float(np.mean(train_times)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
    }

    print(f"\n=== Scratch NN | Digits | {training_percent}% of training data ===")
    print(f"Mean training time: {results['mean_train_time']:.3f} s")
    print(f"Mean accuracy:      {results['mean_accuracy']*100:.2f}%")
    print(f"Mean error:         {results['mean_error']*100:.2f}%")
    print(f"Std of error:       {results['std_error']*100:.2f}%")
    return results


if __name__ == "__main__":
    percent = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(percent)