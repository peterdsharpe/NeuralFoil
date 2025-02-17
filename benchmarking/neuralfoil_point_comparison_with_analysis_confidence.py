import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools.string_formatting import eng_string
from neuralfoil import get_aero_from_airfoil
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

af = asb.Airfoil(
    name="HALE_03", coordinates=Path(__file__).parent / "assets" / "hale_03mod.dat"
).to_kulfan_airfoil()
alphas_xfoil = np.linspace(-5, 15, 50)
alphas_nf = np.linspace(-6, 17, 300)
Re_values_to_test = [1e4, 8e4, 2e5, 1e6, 1e8]

# Obtain data
aeros = {}

aeros["xfoil"] = [
    asb.XFoil(
        airfoil=af,
        Re=Re,
        timeout=30,
        # xfoil_repanel=False,
        max_iter=100,
    ).alpha(alphas_xfoil)
    for Re in tqdm(Re_values_to_test, desc="XFoil")
]

nf_model_sizes = ["xxxlarge"]
nf_linestyles = ["-"]
xfoil_linestyle = "k:"

for model_size in nf_model_sizes:
    aeros[model_size] = [
        get_aero_from_airfoil(airfoil=af, alpha=alphas_nf, Re=Re, model_size=model_size)
        for Re in tqdm(Re_values_to_test, desc=f"NeuralFoil {model_size}")
    ]

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots(figsize=(7, 5))
plt.xscale("log")

cmap = LinearSegmentedColormap.from_list(
    "custom_cmap",
    colors=[
        p.adjust_lightness(c, 0.8) for c in ["orange", "darkseagreen", "dodgerblue"]
    ],
)
colors = cmap(np.linspace(0, 1, len(Re_values_to_test)))
transparency = 0.7

for i, (Re, color) in enumerate(zip(Re_values_to_test, colors)):

    for model_size, linestyle in zip(nf_model_sizes, nf_linestyles):
        nf_aero = aeros[model_size][i]
        # nf_line2d, = plt.plot(
        #     nf_aero["CD"],
        #     nf_aero["CL"],
        #     linestyle,
        #     zorder=4,
        #     color=color,
        #     alpha=transparency,
        # )
        lines, sm, cbar = p.plot_color_by_value(
            nf_aero["CD"],
            nf_aero["CL"],
            c=nf_aero["analysis_confidence"],
            clim=(0.8, 1),
            # cmap="turbo_r",
            # cmap="Spectral",
            # cmap="rainbow_r",
            cmap=LinearSegmentedColormap.from_list(
                "custom_cmap",
                colors=[
                    p.adjust_lightness("orange", 1.2),
                    # p.adjust_lightness("gray", 1.0),
                    p.adjust_lightness("darkseagreen", 0.9),
                    p.adjust_lightness("dodgerblue", 0.6),
                ],
            ),
            alpha=transparency,
            zorder=6,
        )

    (xfoil_line2d,) = plt.plot(
        aeros["xfoil"][i]["CD"],
        aeros["xfoil"][i]["CL"],
        linestyle=(0, (1, 1.5)),
        linewidth=2.2,
        # ".", markeredgewidth=0, markersize=4, alpha=0.8,
        color="k",
        zorder=5,
    )

    annotate_x = np.max(np.array([aero[i]["CD"][-1] for aero in aeros.values()]))
    annotate_y = np.median(np.array([aero[i]["CL"][-1] for aero in aeros.values()]))

    plt.annotate(
        f" ${{\\rm Re}} = \\mathrm{{{eng_string(Re)}}}$",
        xy=(annotate_x, annotate_y),
        # color=p.adjust_lightness(color, 0.8),
        color="k",
        ha="left",
        va="center",
        fontsize=10,
    )

plt.colorbar(
    sm,
    ax=plt.gca(),
    label="Analysis Confidence",
    pad=0.063,
)

plt.annotate(
    text="Note the log-scale on $C_D$, which is unconventional - this\nis used to keep $C_D$ readable given the wide range.",
    xy=(0.01, 0.01),
    xycoords="figure fraction",
    ha="left",
    va="bottom",
    fontsize=8,
    alpha=0.5,
)

from matplotlib.lines import Line2D

legend_handles = [
    Line2D(
        [],
        [],
        color="k",
        # linestyle=xfoil_line2d.get_linestyle(),
        linestyle=(0, (1, 1.5)),
        linewidth=2.2,
        label="XFoil (ground truth)",
    ),
    *[
        Line2D([], [], color=color, linestyle=linestyle, label=f'NF "{model_size}"')
        for model_size, linestyle, color in zip(nf_model_sizes, nf_linestyles, colors)
    ],
]

for h in legend_handles:
    h.set_color("k")

plt.legend(
    handles=legend_handles,
    title="Analysis Method",
    # loc=(0.8, 0.15),
    loc="lower left",
    # bbox_to_anchor=(0.5, 0., 0.5, 0.5),
    fontsize=11,
    labelspacing=0.3,
    columnspacing=1.5,
    handletextpad=0.4,
    framealpha=0.8,
)

plt.xlim()
plt.ylim(bottom=-0.8)

afax = ax.inset_axes([0.76, 0.802, 0.23, 0.23])
afax.fill(
    af.x(), af.y(), facecolor=(0, 0, 0, 0.2), linewidth=1, edgecolor=(0, 0, 0, 0.7)
)
afax.annotate(
    text=f"{af.name} Airfoil",
    xy=(0.5, 0.15),
    xycoords="data",
    ha="center",
    va="bottom",
    fontsize=10,
    alpha=0.7,
)

afax.grid(False)
afax.set_xticks([])
afax.set_yticks([])
# afax.axis('off')
afax.set_facecolor((1, 1, 1, 0.5))
afax.set_xlim(-0.05, 1.05)
afax.set_ylim(-0.05, 0.28)
afax.set_aspect("equal", adjustable="box")


plt.suptitle("Comparison of $C_L$-$C_D$ Polar for NeuralFoil vs. XFoil", fontsize=16)
plt.title(f"On {af.name} Airfoil (out-of-sample)", fontsize=12, alpha=0.7)

plt.xlabel("Drag Coefficient $C_D$")
plt.ylabel("Lift Coefficient $C_L$")

p.show_plot(
    None,
    legend=False,
    savefig="neuralfoil_point_comparison_with_analysis_confidence.svg",
    savefig_transparent=False,
    rotate_axis_labels=False,
)

# p.plot_color_by_value(
#     nf_aero_xl["CD"],
#     nf_aero_xl["CL"],
#     c=nf_aero_xl["analysis_confidence"],
#     clim=(0.8, 1),
#     colorbar=Re == Re_values_to_test[-1],
#     colorbar_label="Analysis Confidence" if Re == Re_values_to_test[-1] else None,
#     cmap="turbo_r",
#     # "--",
#     # color=color,
#     zorder=4,
# )

# plt.plot(
#     nf_aero_xl["CD"],
#     nf_aero_xl["CL"],
#     "--",
#     color=color, alpha=transparency
# )

# nf_aero_m = get_aero_from_airfoil(
#     airfoil=af,
#     alpha=alphas_nf,
#     Re=Re,
#     model_size="medium"
# )
#
# plt.plot(
#     nf_aero_m["CD"],
#     nf_aero_m["CL"],
#     ":",
#     color=color, alpha=transparency
# )

# xfoil_aero = asb.XFoil(
#     airfoil=af,
#     Re=Re,
#     timeout=30,
#     max_iter=100,
# ).alpha(alpha=alphas_xfoil, start_at=5)
#
# plt.plot(
#     xfoil_aero["CD"],
#     xfoil_aero["CL"],
#     "--",
#     color="k", alpha=0.4
# )
# xfoil_aero = nf_aero
