#!/usr/bin/env python3
import argparse
import datetime
import os
import re

os.environ.setdefault(
    "KERAS_BACKEND", "torch"
)  # Use PyTorch backend unless specified otherwise

import keras
import numpy as np
import torch
import torch.utils.tensorboard

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument(
    "--hidden_layer", default=100, type=int, help="Size of the hidden layer."
)
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate.")
parser.add_argument(
    "--recodex", default=False, action="store_true", help="Evaluation in ReCodEx."
)
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(keras.Model):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self._args = args

        self._W1 = keras.Variable(
            keras.random.normal(
                [MNIST.W * MNIST.H * MNIST.C, args.hidden_layer],
                stddev=0.1,
                seed=args.seed,
            ),
            trainable=True,
        )
        self._b1 = keras.Variable(keras.ops.zeros([args.hidden_layer]), trainable=True)

        self._W2 = keras.Variable(
            keras.random.normal(
                [args.hidden_layer, MNIST.LABELS], stddev=0.1, seed=args.seed
            ),
            trainable=True,
        )
        self._b2 = keras.Variable(keras.ops.zeros([MNIST.LABELS]), trainable=True)

    def predict(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = keras.ops.cast(inputs, dtype="float32")
        inputs = inputs / 255.0
        inputs = inputs.reshape([inputs.shape[0], -1])

        first_layer_output = keras.ops.tanh(inputs @ self._W1 + self._b1)

        output = keras.ops.softmax(first_layer_output @ self._W2 + self._b2)

        # In order to support manual gradient computation, you should
        # return not only the output layer, but also the hidden layer after applying
        # tanh, and the input layer after reshaping.
        return output, first_layer_output, inputs

    def train_epoch(self, dataset) -> None:
        for batch in dataset.batches(self._args.batch_size):
            # The batch contains
            # - batch["images"] with shape [?, MNIST.H, MNIST.W, MNIST.C]
            # - batch["labels"] with shape [?]
            # Size of the batch is `self._args.batch_size`, except for the last, which
            # might be smaller.

            # Contrary to `sgd_backpropagation`, the goal here is to compute
            # the gradient manually, without calling `.backward()`. ReCodEx disables
            # PyTorch automatic differentiation during evaluation.
            #
            # Compute the input layer, hidden layer and output layer
            # of the batch images using `self.predict`.

            output, hidden_layer, inputs = self.predict(batch["images"])

            labels_one_hot = keras.ops.one_hot(batch["labels"], MNIST.LABELS)

            output_der = output - labels_one_hot

            b2_grad = output_der.mean(0)
            w2_grad = keras.ops.einsum("ai,aj->aij", hidden_layer, output_der).mean(0)

            hidden_der = 1 - hidden_layer**2

            hidden_grad = output_der @ torch.transpose(self._W2, 0, 1) * hidden_der
            b1_grad = hidden_grad.mean(0)

            w1_grad = keras.ops.einsum("ai,aj->aij", inputs, hidden_grad).mean(0)

            # Compute the gradient of the loss with respect to all
            # variables. Note that the loss is computed as in `sgd_backpropagation`:
            # - For every batch example, the loss is the categorical crossentropy of the
            #   predicted probabilities and the gold label. To compute the crossentropy, you can
            #   - either use `keras.ops.one_hot` to obtain one-hot encoded gold labels,
            #   - or suitably use `keras.ops.take_along_axis` to "index" the predicted probabilities.
            # - Finally, compute the average across the batch examples.
            #
            # During the gradient computation, you will need to compute
            # a batched version of a so-called outer product
            #   `C[a, i, j] = A[a, i] * B[a, j]`,
            # which you can achieve by using for example
            #   `A[:, :, np.newaxis] * B[:, np.newaxis, :]`
            # or with
            #   `keras.ops.einsum("ai,aj->aij", A, B)`.

            # Perform the SGD update with learning rate `self._args.learning_rate`
            # for the variable and computed gradient. You can modify the
            # variable value with `variable.assign` or in this case the more
            # efficient `variable.assign_sub`.

            ##Update biases
            self._b1.assign(self._b1 - b1_grad * self._args.learning_rate)
            self._b2.assign(self._b2 - b2_grad * self._args.learning_rate)

            ##Update weights
            self._W1.assign(self._W1 - w1_grad * self._args.learning_rate)
            self._W2.assign(self._W2 - w2_grad * self._args.learning_rate)

    # def train_epoch(self, dataset) -> None:
    #     for batch in dataset.batches(self._args.batch_size):
    #         # The batch contains
    #         # - batch["images"] with shape [?, MNIST.H, MNIST.W, MNIST.C]
    #         # - batch["labels"] with shape [?]
    #         # Size of the batch is `self._args.batch_size`, except for the last, which
    #         # might be smaller.

    #         # Contrary to `sgd_backpropagation`, the goal here is to compute
    #         # the gradient manually, without calling `.backward()`. ReCodEx disables
    #         # PyTorch automatic differentiation during evaluation.
    #         #
    #         # Compute the input layer, hidden layer, and output layer
    #         # of the batch images using `self.predict`.

    #         output, hidden_layer, inputs = self.predict(batch["images"])

    #         labels_one_hot = keras.ops.one_hot(batch["labels"], MNIST.LABELS)

    #         output_der = output - labels_one_hot

    #         b2_grad = output_der.mean(dim=0)
    #         w2_grad = hidden_layer.transpose(0, 1) @ output_der

    #         hidden_der = 1 - hidden_layer**2

    #         hidden_grad = (output_der @ torch.transpose(self._W2, 0, 1)) * hidden_der

    #         b1_grad = hidden_grad.mean(dim=0)
    #         w1_grad = inputs.transpose(0, 1) @ hidden_grad

    #         # Update biases
    #         self._b1.assign_sub(b1_grad * self._args.learning_rate)
    #         self._b2.assign_sub(b2_grad * self._args.learning_rate)

    #         # Update weights
    #         self._W1.assign_sub(w1_grad * self._args.learning_rate)
    #         self._W2.assign_sub(w2_grad * self._args.learning_rate)

    def evaluate(self, dataset) -> float:
        # Compute the accuracy of the model prediction
        correct = 0
        for batch in dataset.batches(self._args.batch_size):
            ##Extract only the output values (discard outputs of other layers)
            (
                raw,
                _,
                _,
            ) = self.predict(batch["images"])
            probabilities = keras.ops.convert_to_numpy(raw)
            predicted_labels = np.argmax(probabilities, axis=1)

            correct += np.sum(predicted_labels == batch["labels"])

        return correct / dataset.size


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join(
        "logs",
        "{}-{}-{}".format(
            os.path.basename(globals().get("__file__", "notebook")),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(
                (
                    "{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v)
                    for k, v in sorted(vars(args).items())
                )
            ),
        ),
    )

    # Load data
    mnist = MNIST()

    # Create the TensorBoard writer
    writer = torch.utils.tensorboard.SummaryWriter(args.logdir)

    # Create the model
    model = Model(args)

    for epoch in range(args.epochs):
        # Run the `train_epoch` with `mnist.train` dataset
        model.train_epoch(dataset=mnist.train)

        # Evaluate the dev data using `evaluate` on `mnist.dev` dataset
        accuracy = model.evaluate(dataset=mnist.dev)
        print(
            "Dev accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * accuracy),
            flush=True,
        )
        writer.add_scalar("dev/accuracy", 100 * accuracy, epoch + 1)

    # Evaluate the test data using `evaluate` on `mnist.test` dataset
    test_accuracy = model.evaluate(dataset=mnist.test)
    print(
        "Test accuracy after epoch {} is {:.2f}".format(epoch + 1, 100 * test_accuracy),
        flush=True,
    )
    writer.add_scalar("test/accuracy", 100 * test_accuracy, epoch + 1)

    # Return dev and test accuracies for ReCodEx to validate.
    return accuracy, test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
