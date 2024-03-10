from pathlib import Path
from typing import List
import numpy as np
from scipy import interpolate, ndimage
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

# log_files = {
#     f.name: int(f.suffix.split("-")[-1])
#     for f in Path(__file__).parent.glob("*.log-*")
# }

# get dictionary key where value is highest
# log_file = max(log_files, key=log_files.get)

log_file_ids = {
    "xxsmall" : "log.log-25362570",
    "xsmall"  : "log.log-25362571",
    "small"   : "log.log-25362572",
    "medium"  : "log.log-25362575",
    "large"   : "log.log-25362579",
    "xlarge"  : "log.log-25362580",
    "xxlarge" : "log.log-25362581",
    "xxxlarge": "log.log-25362584",
}
# log_file_ids = {
#     "1e-6": "log.log-25362440",
#     "1e-5": "log.log-25362441",
#     "3e-5": "log.log-25362442",
#     "1e-4": "log.log-25362443",
#     "3e-4": "log.log-25362444",
#     "1e-3": "log.log-25362445",
#     "1e-2": "log.log-25362448",
#     "1e-1": "log.log-25362449",
# }

fig, ax = plt.subplots(
    ncols=int(np.ceil(len(log_file_ids) / 2)),
    nrows=2,
    figsize=(12, 8),
    sharex=True,
    sharey=True,
)
ax_f = ax.flatten()

for i, (title, log_file) in enumerate(log_file_ids.items()):

    with open(Path(__file__).parent / log_file, "r", encoding="utf8") as f:
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
        if line.startswith("Epoch: "):
            row = parse_line(line)
            if row is not None:
                for k, v in row.items():
                    if k not in data:
                        data[k] = []
                    data[k].append(v)

    data = {k: np.array(v) for k, v in data.items()}

    plt.sca(ax_f[i])

    for key in ["Train Loss", "Test Loss"]:
        lines = plt.plot(
            data[key],
            ".-",
            markersize=5,
            alpha=0.5,
            markeredgewidth=0,
        )

        # plt.plot(
        #     np.exp(ndimage.gaussian_filter1d(np.log(data[key]), sigma=0.1)),
        #     color=lines[0].get_color(), alpha=0.7,
        #     label=key, zorder=4
        # )

    plt.yscale('log')
    # plt.ylim(2e-3, 2e-2)
    plt.title(f"{title}, {log_file}\nCD {np.median(data['ln_CD'][-10:]):.4f}")

p.show_plot(f"Training Progress", "Epoch", "Loss")

# print(np.exp(ndimage.gaussian_filter1d(np.log(data[key]), sigma=10)[-1]))
# from pprint import pprint
#
# pprint({k: v[-1] for k, v in data.items()})
