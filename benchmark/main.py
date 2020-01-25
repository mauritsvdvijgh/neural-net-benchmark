from argparse import ArgumentParser
import logging
import ujson as json
import time
import pickle
import requests


AVAILABLE_FRAMEWORKS = ["scikit", "tensorflow", "cntk", "theano", "mxnet", "pytorch"]
AVAILABLE_DATASETS = ["mnist"]
AVAILABLE_MODES = ["http", "direct", "celery"]


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


def run_benchmark_http(framework, data, epochs=1, split=1):
    x_train, y_train, x_test, y_test = data
    url = "http://{}:8000".format(framework)
    test_data = json.dumps({"samples": x_test.tolist()})
    step_size = len(x_train) // split
    stats = {"step_size": step_size, "framework": framework, "epochs": {}}
    model_config = {"hidden_layer_sizes": (100, ), "classes" : [0,1,2,3,4,5,6,7,8,9], "input_dim" : 784}
    requests.put(url, json=model_config)

    for epoch in range(epochs):
        stats["epochs"][epoch] = {"mem": [], "time": [], "accuracy": []}
        epoch_stats = stats["epochs"][epoch]
        for step in range(0, len(x_train), step_size):
            train_data = json.dumps(
                {
                    "samples": [x_train[step : step + step_size].tolist()],
                    "classes": y_train[step : step + step_size].tolist(),
                }
            )
            train_response = requests.post(url, data=train_data).json()
            test_response = requests.get(url, data=test_data).json()
            accuracy = get_accuracy(y_test, test_response["classes"])
            epoch_stats["time"].append(train_response["time"])
            epoch_stats["mem"].append(train_response["mem"])
            epoch_stats["accuracy"].append(accuracy)
    return stats


def run_benchmark_direct(data, epochs=1, split=1):
    x_train, y_train, x_test, y_test = data
    url = "http://{}:8000".format(framework)
    test_data = json.dumps({"samples": x_test.tolist()})
    step_size = len(x_train) // split
    stats = {"step_size": step_size, "framework": framework, "epochs": {}}
    model_config = {"hidden_layer_sizes": (100, ), "classes" : [0,1,2,3,4,5,6,7,8,9], "input_dim" : 784}
    requests.put(url, json=model_config)

    for epoch in range(epochs):
        stats["epochs"][epoch] = {"mem": [], "time": [], "accuracy": []}
        epoch_stats = stats["epochs"][epoch]
        for step in range(0, len(x_train), step_size):
            train_data = json.dumps(
                {
                    "samples": [x_train[step : step + step_size].tolist()],
                    "classes": y_train[step : step + step_size].tolist(),
                }
            )
            train_response = requests.post(url, data=train_data).json()
            test_response = requests.get(url, data=test_data).json()
            accuracy = get_accuracy(y_test, test_response["classes"])
            epoch_stats["time"].append(train_response["time"])
            epoch_stats["mem"].append(train_response["mem"])
            epoch_stats["accuracy"].append(accuracy)
    return stats


def summarize(stats_run):
    return "{} | epochs: {} | batch_size: {} | training time: {} | accuracy: {}".format(
        stats_run["framework"],
        len(stats_run["epochs"]),
        stats_run["step_size"],
        sum(
            [
                time
                for epoch_stats in stats_run["epochs"]
                for time in stats_run["epochs"][epoch_stats]["time"]
            ]
        ),
        stats_run["epochs"][max(stats_run["epochs"])]["accuracy"][-1],
    )


def main(frameworks, datasets, epochs, splits):
    stats = []
    for data in datasets:
        for framework in frameworks:
            for split in splits:
                data = load_mnist()
                stats_run = run_benchmark(framework, data, epochs, split)
                stats.append(stats_run)
                print(summarize(stats_run))
    print(stats)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="selected_mode",
        type: str,
        default='http'
        help="benchmark mode",
    )

    parser.add_argument(
        "--frameworks",
        dest="selected_frameworks",
        type=lambda s: [str(item) for item in s.split(",")],
        default=AVAILABLE_FRAMEWORKS,
        help="frameworks to benchmark",
    )
    parser.add_argument(
        "--epochs", dest="epochs", type=int, default=1, help="epochs to train"
    )
    parser.add_argument(
        "--splits",
        dest="splits",
        type=lambda s: [int(item) for item in s.split(",")],
        default=[1],
        help="number of parts to split the dataset in before training",
    )
    parser.add_argument(
        "--datasets",
        dest="selected_datasets",
        type=lambda s: [str(item) for item in s.split(",")],
        default=AVAILABLE_DATASETS,
        help="datasets to benchmark",
    )
    args = parser.parse_args()
    main(args.selected_frameworks, args.selected_datasets, args.epochs, args.splits)
