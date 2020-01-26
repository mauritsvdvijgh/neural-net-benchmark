# -*- coding: utf-8 -*-
import fileinput
import json

import plotly
import plotly.graph_objects as go


plotly.io.orca.config.server_url = "localhost:9091"
first = True
framework = None
fig = go.Figure()
x, y, point_labels = [], [], []


for line in fileinput.input():
    data = json.loads(line)
    total_time = sum(data["times"])
    predictions = len(data["times"])
    print(data["framework"], "&", predictions / total_time, "&", total_time, "\\\\")
