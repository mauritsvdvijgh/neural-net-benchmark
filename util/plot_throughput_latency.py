# -*- coding: utf-8 -*-
import fileinput
import json
import os
import sys

import plotly
import plotly.graph_objects as go


plotly.io.orca.config.server_url = "orca:9091"
first = True
framework = None
fig = go.Figure()
x, y, point_labels = [], [], []
for line in fileinput.input():
    data = json.loads(line)

    times = [
        time
        for epoch_stats in data["epochs"]
        for time in data["epochs"][epoch_stats]["time"]
    ]

    batch_size = data["batch_size"]
    epochs = len(data["epochs"])
    total_time = sum(times)
    points_per_second = 60000 / total_time
    latency = total_time / len(times)
    accuracy = data["epochs"][max(data["epochs"])]["accuracy"]

    if data["framework"] != framework:
        if framework:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    text=point_labels,
                    mode="markers",
                    marker=dict(size=12),
                    name=framework,
                )
            )
            x, y, point_labels = [], [], []
        framework = data["framework"]

    x.append(points_per_second)
    y.append(1000 * latency)
    point_labels.append(f"acc: {accuracy} | batch size: {batch_size}")

fig.add_trace(
    go.Scatter(
        x=x,
        y=y,
        text=point_labels,
        mode="markers",
        marker=dict(size=12),
        name=framework,
    )
)

fig.update_layout(
    font=dict(size=18),
    yaxis=dict(title="Latency (ms)"),
    xaxis=dict(title="Training speed (samples/sec)"),
)

if not os.isatty(sys.stdout.fileno()):
    sys.stdout.buffer.write(fig.to_image(format="pdf"))
else:
    fig.show()
