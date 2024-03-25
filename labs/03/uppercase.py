#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import numpy as np

os.environ.setdefault(
    "KERAS_BACKEND", "torch"
)  # Use PyTorch backend unless specified otherwise

import keras
import torch

from uppercase_data import UppercaseData

# TODO: Set reasonable values for the hyperparameters, especially for
# `alphabet_size`, `batch_size`, `epochs`, and `window`.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--alphabet_size",
    default=110,
    type=int,
    help="If given, use this many most frequent chars.",
)
parser.add_argument("--batch_size", default=500, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument(
    "--threads", default=1, type=int, help="Maximum number of threads to use."
)
parser.add_argument("--window", default=3, type=int, help="Window size to use.")


class TorchTensorBoardCallback(keras.callbacks.Callback):
    def __init__(self, path):
        self._path = path
        self._writers = {}

    def writer(self, writer):
        if writer not in self._writers:
            import torch.utils.tensorboard

            self._writers[writer] = torch.utils.tensorboard.SummaryWriter(
                os.path.join(self._path, writer)
            )
        return self._writers[writer]

    def add_logs(self, writer, logs, step):
        if logs:
            for key, value in logs.items():
                self.writer(writer).add_scalar(key, value, step)
            self.writer(writer).flush()

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            if isinstance(
                getattr(self.model, "optimizer", None), keras.optimizers.Optimizer
            ):
                logs = logs | {
                    "learning_rate": keras.ops.convert_to_numpy(
                        self.model.optimizer.learning_rate
                    )
                }
            self.add_logs(
                "train",
                {k: v for k, v in logs.items() if not k.startswith("val_")},
                epoch + 1,
            )
            self.add_logs(
                "val",
                {k[4:]: v for k, v in logs.items() if k.startswith("val_")},
                epoch + 1,
            )


def main(args: argparse.Namespace) -> None:
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
    uppercase_data = UppercaseData(args.window, args.alphabet_size)

    # print(uppercase_data.train.data)

    # TODO: Implement a suitable model, optionally including regularization, select
    # good hyperparameters and train the model.
    #
    # The inputs are _windows_ of fixed size (`args.window` characters on the left,
    # the character in question, and `args.window` characters on the right), where
    # each character is represented by a "int32" index. To suitably represent
    # the characters, you can:
    # - Convert the character indices into _one-hot encoding_, which you can achieve by
    #   - suitable application of the `layers.CategoryEncoding` layer
    #   - when using Functional API, any `keras.ops` can be used as a Keras layer:
    #       inputs = keras.layers.Input(shape=[2 * args.window + 1], dtype="int32")
    #       encoded = keras.ops.one_hot(inputs, len(uppercase_data.train.alphabet))
    #   You can then flatten the one-hot encoded windows and follow with a dense layer.
    # - Alternatively, you can use `keras.layers.Embedding` (which is an efficient
    #   implementation of one-hot encoding followed by a Dense layer) and flatten afterwards.

    ##Choice of constants
    num_models = 5
    models = []

    hidden_layer_1 = 200
    hidden_layer_2 = 200
    for i in range(num_models):
        model = keras.Sequential()
        model.add(keras.Input([2 * args.window + 1]))
        model.add(keras.layers.Embedding(args.alphabet_size, 100))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(hidden_layer_1, activation="elu"))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(hidden_layer_2, activation="elu"))
        model.add(keras.layers.Dropout(0.1))
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        model.summary()

        model.compile(
            optimizer=keras.optimizers.AdamW(),
            loss=keras.losses.BinaryCrossentropy(label_smoothing=0.01),
            metrics=[keras.metrics.BinaryAccuracy("accuracy")],
        )

        model.fit(
            uppercase_data.train.data["windows"],
            uppercase_data.train.data["labels"],
            batch_size=args.batch_size,
            epochs=args.epochs,
            validation_data=(
                uppercase_data.dev.data["windows"],
                uppercase_data.dev.data["labels"],
            ),
        )
        models.append(model)

    # TODO: Generate correctly capitalized test set.
    # Use `uppercase_data.test.text` as input, capitalize suitable characters,
    # and write the result to predictions_file (which is
    # `uppercase_test.txt` in the `args.logdir` directory).

    def capitalize_with_boolean_array(input_string, capitalize_array):
        capitalized_string = ""
        for char, capitalize_flag in zip(input_string, capitalize_array):
            if capitalize_flag >= 0.5:
                capitalized_string += char.upper()
            else:
                capitalized_string += char
        return capitalized_string

    # prediction_val = model.predict(uppercase_data.dev.data["windows"])

    preds = [mod.predict(uppercase_data.test.data["windows"]) for mod in models]
    avg_pred = np.mean(preds, axis=0)

    os.makedirs(args.logdir, exist_ok=True)
    with open(
        os.path.join(args.logdir, "uppercase_test.txt"), "w", encoding="utf-8"
    ) as predictions_file:
        prediction = preds
        predictions_file.write(
            capitalize_with_boolean_array(uppercase_data.test.text, avg_pred)
        )


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
