# -*- coding: utf-8 -*-
import inspect
import json
import logging
import os
import pickle
import resource
import time
from argparse import ArgumentParser

import models
import numpy as np


AVAILABLE_MODELS = {
    model[0].lower(): model[1] for model in inspect.getmembers(models, inspect.isclass)
}
MODEL_CLASS = AVAILABLE_MODELS[os.environ.get("FRAMEWORK", "scikit")]

# init model with http put
model = None


def init(hidden_layer_sizes, classes, input_dim, batch_size):
    global model
    model = MODEL_CLASS(hidden_layer_sizes, classes, input_dim, batch_size=batch_size)


def train(samples, labels):
    start = time.time()
    model.train(samples, labels)
    end = time.time()
    return {
        "time": end - start,
        "mem": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def predict(samples):
    start = time.time()
    predictions = model.predict(samples)
    end = time.time()
    return {"classes": predictions.tolist(), "time": end - start}


def load_mnist():
    with open("/data/mnist.pkl", "rb") as f:
        mnist = pickle.load(f)
    return (
        mnist["training_images"],
        mnist["training_labels"],
        mnist["test_images"],
        mnist["test_labels"],
    )


def get_accuracy(answers, predictions):
    right = sum(1 for a, p in zip(answers, predictions) if a == p)
    total = len(answers)
    return right / total


def summarize(stats_run):
    return "{} | epochs: {} | batch_size: {} | training time: {} | accuracy: {}".format(
        stats_run["framework"],
        len(stats_run["epochs"]),
        stats_run["batch_size"],
        sum(
            [
                time
                for epoch_stats in stats_run["epochs"]
                for time in stats_run["epochs"][epoch_stats]["time"]
            ]
        ),
        stats_run["epochs"][max(stats_run["epochs"])]["accuracy"],
    )


def run_benchmark_direct(data, epochs=1, batch_size=60000):
    x_train, y_train, x_test, y_test = data
    init((200, 200), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 784, min(200, batch_size))
    stats = {
        "batch_size": batch_size,
        "framework": os.environ.get(
            "KERAS_BACKEND", os.environ.get("FRAMEWORK", "scikit")
        ),
        "epochs": {},
    }
    for epoch in range(epochs):
        stats["epochs"][epoch] = {"mem": [], "time": []}
        epoch_stats = stats["epochs"][epoch]

        for batch_offset in range(0, len(x_train) - 1, batch_size):
            train_feedback = train(
                x_train[batch_offset : batch_offset + batch_size],
                y_train[batch_offset : batch_offset + batch_size],
            )
            epoch_stats["time"].append(train_feedback["time"])
            epoch_stats["mem"].append(train_feedback["mem"])

        test_feedback = predict(x_test)
        accuracy = get_accuracy(y_test, test_feedback["classes"])
        epoch_stats["accuracy"] = accuracy
    return stats


def run_benchmark_predict(data):
    x_train, y_train, x_test, y_test = data
    stats = []
    for i in range(0, len(x_test), 1):
        start = time.time()
        predict(x_test[i : i + 1])
        end = time.time()
        stats.append(end - start)
    return stats


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--predict", dest="predict", action="store_true")
    parser.add_argument(
        "--epochs", dest="epochs", type=int, default=1, help="epochs to train"
    )
    parser.add_argument(
        "--batch_size",
        dest="selected_batch_size",
        type=int,
        default=1,
        help="batch size",
    )
    args = parser.parse_args()

    data = load_mnist()
    stats = run_benchmark_direct(data, args.epochs, args.selected_batch_size)
    if args.predict:
        stats = {
            "times": run_benchmark_predict(data),
            "framework": os.environ.get(
                "KERAS_BACKEND", os.environ.get("FRAMEWORK", "scikit")
            ),
        }
    print(json.dumps(stats))
