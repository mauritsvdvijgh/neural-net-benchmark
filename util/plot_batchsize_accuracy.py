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
batch_sizes = set()
for line in fileinput.input():
    data = json.loads(line)

    times = [
        time
        for epoch_stats in data["epochs"]
        for time in data["epochs"][epoch_stats]["time"]
    ]

    batch_size = data["batch_size"]
    batch_sizes.add(batch_size)
    epochs = len(data["epochs"])
    total_time = sum(times)
    points_per_second = (epochs*60000) / total_time
    latency = total_time / len(times)
    accuracy = data["epochs"][max(data["epochs"])]["accuracy"]

    if data["framework"] != framework:
        if framework:
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=y,
                    text=point_labels,
                    name=framework,
                )
            )
            x, y, point_labels = [], [], []
        framework = data["framework"]

    x.append(batch_size)
    y.append(accuracy)
    point_labels.append(f"acc: {accuracy} | batch size: {batch_size}")

fig.add_trace(
    go.Bar(
        x=x,
        y=y,
        text=point_labels,
        name=framework,
    )
)

fig.update_layout(
    font=dict(size=18),
    yaxis=dict(title="Accuracy"),
    xaxis=dict(title="Batch size", type='category', categoryorder='array', categoryarray=sorted(batch_sizes)),
)

if not os.isatty(sys.stdout.fileno()):
    sys.stdout.buffer.write(fig.to_image(format="pdf"))
else:
    fig.show()
