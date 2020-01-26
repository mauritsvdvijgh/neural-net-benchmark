# -*- coding: utf-8 -*-
import fileinput
import json


for line in fileinput.input():
    data = json.loads(line)
    total_time = sum(data["times"])
    predictions = len(data["times"])
    print(data["framework"], "&", predictions / total_time, "&", total_time, "\\\\")
