import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
from sortedcontainers import SortedDict
import pandas as pd
from pathlib import Path
from typing import Any

data = Path(__file__).parent / "data"

# Load reference data
reference_data = pd.read_csv(data / "reference_data.csv")

# Load XFoil data
xfoil_data = SortedDict()
for file in data.glob("xfoil_*.csv"):
    N = int(file.stem.split("_")[1])
    xfoil_data[N] = pd.read_csv(file)

# Load NeuralFoil data
nf_data = {}
for file in data.glob("nf_*.csv"):
    model_size = file.stem.split("_")[1]
    nf_data[model_size] = pd.read_csv(file)
nf_data = {  # Sorts the dictionary
    k: nf_data[k]
    for k in ["xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge"]
}

# Load Vectorized NeuralFoil data
vect_nf_data = {}
for file in data.glob("vect_nf_*.csv"):
    model_size = file.stem.split("_")[2]
    vect_nf_data[model_size] = pd.read_csv(file)
vect_nf_data = {  # Sorts the dictionary
    k: vect_nf_data[k]
    for k in ["xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge"]
}


def parse_data_time_and_error(data: pd.DataFrame) -> tuple[float, float]:
    error = np.nanmean(np.abs(np.log(data["CD"]) - np.log(reference_data["CD"])))
    runtime = np.median(data["time"])
    return error, runtime


def parse_data_series_time_and_error(data: dict[Any, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray]:
    outs: list[tuple[float, float]] = [
        parse_data_time_and_error(df)
        for df in data.values()
    ]
    errors, runtimes = zip(*outs)
    return errors, runtimes


fig, ax = plt.subplots(figsize=(6.5, 4.5))

xf_line, = plt.plot(
    *parse_data_series_time_and_error(xfoil_data),
    "o-",
    label="XFoil", color="k",
)
nf_line, = plt.plot(
    *parse_data_series_time_and_error(nf_data),
    "s-",
    label="NeuralFoil (naïve looping)",
    color="C0"
)
vect_nf_line, = plt.plot(
    *parse_data_series_time_and_error(vect_nf_data),
    "^-",
    label="NeuralFoil (vectorized)",
    color="C1"
)

# Do annotations
def anno(
        **kwargs,
):
    defaults = dict(
        xytext=(0, 20),
        xycoords="data",
        va="center",
        ha="center",
        textcoords="offset points",
        fontsize=8,
        alpha=0.6,
        arrowprops={
            "arrowstyle": "->",
            "color"     : "k",
            "alpha"     : 0.35,
            "shrinkA"   : 0,
            "shrinkB"   : 5
        }
    )
    plt.annotate(**{**defaults, **kwargs})

anno(
    text="$N_{\\rm points}=260$",
    xy=xf_line.get_xydata()[-1],
    xytext=(10, 20),
    va="baseline",
)
anno(
    text="$N_{\\rm points}=20$",
    xy=xf_line.get_xydata()[0],
    xytext=(-10, -20),
    va="top",
)
anno(
    text="\"xxxlarge\"",
    xy=nf_line.get_xydata()[-1],
    xytext=(-10, -15),
    va="top", ha="right",
)
anno(
    text="\"xxsmall\"",
    xy=nf_line.get_xydata()[0],
    xytext=(0, -20),
    va="top",
)
anno(
    text="\"xxxlarge\"",
    xy=vect_nf_line.get_xydata()[-1],
    xytext=(-10, -15),
    va="top", ha="right",
)
anno(
    text="\"xxsmall\"",
    xy=vect_nf_line.get_xydata()[0],
    xytext=(0, 20),
    va="bottom",
)

plt.xscale("log")
plt.yscale("log")

plt.ylim(top=1e0)

# Percentformatter
p.show_plot(show=False)

from matplotlib.ticker import PercentFormatter

ax.xaxis.set_major_formatter(PercentFormatter(1))

p.show_plot(
    "Speed vs. Accuracy Tradeoff for NeuralFoil vs. XFoil",
    "Mean Relative Error of Drag Coefficient $C_D$ [%]",
    "Runtime per Case [sec]",
    set_ticks=False,
    savefig="speed_vs_accuracy_tradeoff.svg"
)

# for N, df in xfoil_data.items():
#     plt.plot(
#         np.median(df["time"]),
#         np.mean(np.abs(df["CD"] - reference_data["CD"])),
#         label=f"XFoil, N={N}",
#         linestyle="-",
#         marker="o",
#         markersize=4,
#     )
