#!/usr/bin/env python3
import argparse
import os
import re

os.environ.setdefault(
    "KERAS_BACKEND", "torch"
)  # Use PyTorch backend unless specified otherwise

import keras
import torch

from mnist import MNIST

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument(
    "--cnn",
    default="CB-8-3-5-valid,R-[CB-8-3-1-same,CB-8-3-1-same],F,H-50",
    type=str,
    help="CNN architecture.",
)
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")
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
        # TODO: Create the model. The template uses the functional API, but
        # feel free to use subclassing if you want.
        inputs = keras.Input(shape=[MNIST.H, MNIST.W, MNIST.C])
        hidden = keras.layers.Rescaling(1 / 255)(inputs)

        def split_string_ignore_square_brackets(input_string):
            pattern = r",(?![^\[\]]*\])"
            return re.split(pattern, input_string)

        def extract_inside_square_brackets(input_string):
            pattern = r"\[([^\[\]]*)\]"
            match = re.search(pattern, input_string)
            if match:
                return match.group(1)
            else:
                return None

        layers = split_string_ignore_square_brackets(args.cnn)
        print(layers)
        # layers = args.cnn.split(",")

        for layer in layers:
            ##Check what layer is to be implemented by checking first letter(s) of the layer string
            splitted_layer = layer.split("-")
            layer_annotation = splitted_layer[0]
            if layer_annotation == "C":
                ##Convolutional layer with ReLU
                hidden = keras.layers.Conv2D(
                    filters=int(splitted_layer[1]),
                    kernel_size=int(splitted_layer[2]),
                    strides=(int(splitted_layer[3]), int(splitted_layer[3])),
                    padding=splitted_layer[4],
                    activation="relu",
                )(hidden)

            elif layer_annotation == "CB":
                # Convolutional layer with batch normalization
                # Has to be split into 3 parts:
                # - Convolutional layer (without activation)
                # - Batch normalization
                # - ReLU activation

                hidden = keras.layers.Conv2D(
                    filters=int(splitted_layer[1]),
                    kernel_size=int(splitted_layer[2]),
                    strides=(int(splitted_layer[3]), int(splitted_layer[3])),
                    padding=splitted_layer[4],
                    activation=None,
                    use_bias=False,
                )(hidden)
                hidden = keras.layers.BatchNormalization()(hidden)
                hidden = keras.layers.Activation("relu")(hidden)

            elif layer_annotation == "M":
                # Max pooling layer
                hidden = keras.layers.MaxPooling2D(
                    pool_size=(int(splitted_layer[1]), int(splitted_layer[1])),
                    strides=int(splitted_layer[2]),
                )(hidden)

            elif layer_annotation == "R":
                # Residual connection
                res_input = hidden
                inner_layers = extract_inside_square_brackets(layer).split(",")
                for inner_layer in inner_layers:
                    inner_layer_split = inner_layer.split("-")
                    if inner_layer_split[0] == "C":
                        hidden = keras.layers.Conv2D(
                            filters=int(inner_layer_split[1]),
                            kernel_size=int(inner_layer_split[2]),
                            strides=int(inner_layer_split[3]),
                            padding=inner_layer_split[4],
                            activation="relu",
                        )(hidden)
                    elif inner_layer_split[0] == "CB":
                        hidden = keras.layers.Conv2D(
                            filters=int(inner_layer_split[1]),
                            kernel_size=int(inner_layer_split[2]),
                            strides=int(inner_layer_split[3]),
                            padding=inner_layer_split[4],
                            activation=None,
                            use_bias=False,
                        )(hidden)
                        hidden = keras.layers.BatchNormalization()(hidden)
                        hidden = keras.layers.Activation("relu")(hidden)

                hidden = keras.layers.Add()([res_input, hidden])

            elif layer_annotation == "F":
                ##Flatten the inputs
                hidden = keras.layers.Flatten()(hidden)

            elif layer_annotation == "H":
                # TODO: Hidden layer
                hidden = keras.layers.Dense(
                    units=int(splitted_layer[1]), activation="relu"
                )(hidden)

            elif layer_annotation == "D":
                # Dropout layer
                hidden = keras.layers.Dropout(rate=float(splitted_layer[1]))(hidden)

        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # a comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer **without bias** and activation,
        #   then add a batch normalization layer, and finally the ReLU activation.
        # - `M-pool_size-stride`: Add max pooling with specified size and stride, using
        #   the default "valid" padding.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the `R` layer should be processed sequentially by `layers`, and the
        #   produced output (after the ReLU nonlinearity of the last layer) should be added
        #   to the input (of this `R` layer).
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `H-hidden_layer_size`: Add a dense layer with ReLU activation and the specified size.
        # - `D-dropout_rate`: Apply dropout with the given dropout rate.
        # You can assume the resulting network is valid; it is fine to crash if it is not.
        #
        # Produce the results in the variable `hidden`.

        # Add the final output layer
        outputs = keras.layers.Dense(MNIST.LABELS, activation="softmax")(hidden)

        super().__init__(inputs=inputs, outputs=outputs)
        self.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    keras.utils.set_random_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    # Load data
    mnist = MNIST()

    # Create the model and train it
    model = Model(args)

    logs = model.fit(
        mnist.train.data["images"],
        mnist.train.data["labels"],
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(mnist.dev.data["images"], mnist.dev.data["labels"]),
    )

    # Return development metrics for ReCodEx to validate.
    return {
        metric: values[-1]
        for metric, values in logs.history.items()
        if metric.startswith("val_")
    }


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
