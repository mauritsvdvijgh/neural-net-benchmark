import os
import time
import inspect
import numpy as np
import falcon
import models
import resource
from argparse import ArgumentParser
import pickle
import logging


AVAILABLE_MODELS = {model[0].lower():model[1] for model in inspect.getmembers(models, inspect.isclass)}
MODEL_CLASS = AVAILABLE_MODELS[os.environ.get('FRAMEWORK', 'scikit')]

# init model with http put
model = None


def init(hidden_layer_sizes, classes, input_dim, batch_size):
    global model
    model = MODEL_CLASS(hidden_layer_sizes, classes, input_dim, batch_size=batch_size)


def train(samples, labels):
    start = time.time()
    model.train(samples, labels)
    end = time.time()
    return {'time': end - start, 'mem': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def predict(samples):
    start = time.time()
    predictions = model.predict(samples)
    end = time.time()
    return {'classes':predictions.tolist(), 'time': end-start}


class ModelResource(object):
    def on_get(self, req, resp):
        try:
            samples = req.media['samples']
        except KeyError:
            resp.media ={}
        else:
            samples = np.array(samples)
            resp.media = predict(samples)

    def on_post(self, req, resp):
        samples = np.array(req.media['samples'][0])
        labels = np.array(req.media['classes'])
        resp.media = train(samples, labels)

    def on_put(self, req, resp):
        init(req.media['hidden_layer_sizes'], req.media['classes'], req.media['input_dim'])


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
    init((100, ), [0,1,2,3,4,5,6,7,8,9], 784, batch_size)
    stats = {"batch_size": batch_size, "framework": os.environ.get('KERAS_BACKEND', os.environ.get('FRAMEWORK', 'scikit')), "epochs": {}}
    for epoch in range(epochs):
        stats["epochs"][epoch] = {"mem": [], "time": []}
        epoch_stats = stats["epochs"][epoch]

        for batch_offset in range(0, len(x_train), batch_size):
            train_feedback = train(x_train[batch_offset:batch_offset+batch_size], y_train[batch_offset:batch_offset+batch_size])
            epoch_stats["time"].append(train_feedback["time"])
            epoch_stats["mem"].append(train_feedback["mem"])

        test_feedback = predict(x_test)
        accuracy = get_accuracy(y_test, test_feedback["classes"])
        epoch_stats["accuracy"] = accuracy
    return stats


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--epochs", dest="epochs", type=int, default=1, help="epochs to train"
    )
    parser.add_argument(
        "--batch_sizes",
        dest="selected_batch_sizes",
        type=lambda s: [int(item) for item in s.split(",")],
        default=[1],
        help="batch sizes",
    )
    args = parser.parse_args()

    stats = []
    data = load_mnist()
    for batch_size in args.selected_batch_sizes:
        stats_run = run_benchmark_direct(data, args.epochs, batch_size)
        stats.append(stats_run)
        print(summarize(stats_run))
    #print(stats)
else:
    app = falcon.API()
    app.add_route('/', ModelResource())
