from pathlib import Path
from typing import List
import numpy as np
from scipy import ndimage

log_files = {
    f.name: int(f.suffix.split("-")[-1]) for f in Path(__file__).parent.glob("*.log-*")
}

# get dictionary key where value is highest
# log_file = max(log_files, key=log_files.get)

log_file = "log.log-23296397"
log_file = "log.log-23296400"
log_file = "log.log-23296405"
log_file = "log.log-23296406"
log_file = "log.log-23296413"
log_file = "log.log-23296435"
log_file = "log.log-23296439"
log_file = "log.log-23296446"

with open(Path(__file__).parent / log_file, "r", encoding="utf8") as f:
    lines = f.readlines()


def parse_line(row):
    kvs: List[str] = row.split("|")
    if len(kvs) != 9:
        return None

    return {kv.split(":")[0].strip(): float(kv.split(":")[1]) for kv in kvs}


data = {}

for line in lines:
    row = parse_line(line)
    if row is not None:
        for k, v in row.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

data = {k: np.array(v) for k, v in data.items()}

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots()

for key in ["Train Loss", "Test Loss"]:
    lines = plt.plot(
        data[key],
        ".",
        markersize=5,
        alpha=0.2,
        markeredgewidth=0,
    )

    plt.plot(
        np.exp(ndimage.gaussian_filter1d(np.log(data[key]), sigma=2)),
        color=lines[0].get_color(),
        alpha=0.7,
        label=key,
        zorder=4,
    )

plt.yscale("log")
plt.ylim(2e-3, 2e-2)
p.show_plot(f"Training Progress ({log_file})", "Epoch", "Loss")

print(np.exp(ndimage.gaussian_filter1d(np.log(data[key]), sigma=10)[-1]))
from pprint import pprint

pprint({k: v[-1] for k, v in data.items()})
