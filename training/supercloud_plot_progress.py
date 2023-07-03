from pathlib import Path
from typing import List
import numpy as np

log_files = {
    f.name: int(f.suffix.split("-")[-1])
    for f in Path(__file__).parent.glob("*.log-*")
}

# get dictioanry key where value is highest
log_file = max(log_files, key=log_files.get)

with open(Path(__file__).parent / log_file, "r") as f:
    lines = f.readlines()

def parse_line(row):
    kvs: List[str] = row.split("|")
    if len(kvs) != 9:
        return None

    return {
        kv.split(":")[0].strip(): float(kv.split(":")[1])
        for kv in kvs
    }

data = {}

for line in lines:
    row = parse_line(line)
    if row is not None:
        for k, v in row.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

data = {k: np.array(v) for k, v in data.items()}
data_epoch = {
    k: np.array([
        np.mean(v[np.array(data["Epoch"]) == epoch])
        for epoch in np.unique(data["Epoch"])
        ])
    for k, v in data.items()
}

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots()
loss = data_epoch["Loss"]
t = np.arange(len(loss))

plt.plot(
    t, loss,
    ".",
    markersize=5,
    alpha=0.8,
    markeredgewidth=0,
)
from scipy import interpolate

plt.plot(
    t, np.exp(interpolate.UnivariateSpline(t, np.log(loss))(t)),
    color="C0", alpha=0.8,
)

plt.yscale('log')
p.show_plot("Training Progress", "Epoch", "Training Loss")