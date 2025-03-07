from pathlib import Path
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

# log_files = {
#     f.name: int(f.suffix.split("-")[-1])
#     for f in Path(__file__).parent.glob("*.log-*")
# }

# get dictionary key where value is highest
# log_file = max(log_files, key=log_files.get)

log_file_ids = {
    "xxsmall": ["log.log-25540774", "log.log-25584785"],
    "xsmall": ["log.log-25542972", "log.log-25584790"],
    "small": "log.log-25542974",
    "medium": "log.log-25542980",
    "large": "log.log-25542983",
    "xlarge": "log.log-25542989",
    "xxlarge": "log.log-25542993",
    "xxxlarge": "log.log-25543001",
}

fig, ax = plt.subplots(
    ncols=int(np.ceil(len(log_file_ids) / 2)),
    nrows=2,
    figsize=(12, 8),
    sharex=True,
    # sharey=True,
)
ax_f = ax.flatten()


def load_log_file(filename) -> dict[str, np.ndarray]:
    with open(filename, "r", encoding="utf8") as f:
        lines = f.readlines()

    def parse_line(row: str) -> Union[dict[str, float], None]:
        kvs: list[str] = row.split("|")
        if len(kvs) != 9:
            return None

        return {kv.split(":")[0].strip(): float(kv.split(":")[1]) for kv in kvs}

    data = {}

    for line in lines:
        if line.startswith("Epoch: "):
            row = parse_line(line)
            if row is not None:
                for k, v in row.items():
                    if k not in data:
                        data[k] = []
                    data[k].append(v)

    return {k: np.array(v) for k, v in data.items()}


for i, (title, log_file_value) in enumerate(log_file_ids.items()):

    if isinstance(log_file_value, str):
        log_file_list = [log_file_value]
    else:
        log_file_list = log_file_value

    datas = [
        load_log_file(Path(__file__).parent / log_file) for log_file in log_file_list
    ]

    data = {k: np.concatenate([d[k] for d in datas]) for k in datas[0].keys()}

    plt.sca(ax_f[i])

    for key in ["Train Loss", "Test Loss"]:
        lines = plt.plot(
            data[key], "-", markersize=5, alpha=0.9, markeredgewidth=0, linewidth=1
        )

        # plt.plot(
        #     np.exp(ndimage.gaussian_filter1d(np.log(data[key]), sigma=0.1)),
        #     color=lines[0].get_color(), alpha=0.7,
        #     label=key, zorder=4
        # )
    plt.ylim(*np.array([0.99, 1.01]) * data["Test Loss"][-1])
    plt.yscale("log")
    # plt.ylim(2e-3, 2e-2)
    plt.title(f"{title}\nCD {np.median(data['ln_CD'][-10:]):.4f}")

p.show_plot("Training Progress", "Epoch", "Loss")

# print(np.exp(ndimage.gaussian_filter1d(np.log(data[key]), sigma=10)[-1]))
# from pprint import pprint
#
# pprint({k: v[-1] for k, v in data.items()})
