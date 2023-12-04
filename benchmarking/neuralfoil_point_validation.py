import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.tools.string_formatting import eng_string
from neuralfoil import get_aero_from_airfoil
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

af = asb.Airfoil(name="HALE_02", coordinates=Path(__file__).parent / "assets" / "HALE_03.dat")

alphas_xfoil = np.linspace(-5, 15, 50)
# alphas_xfoil = np.concatenate([
#     np.sinspace(-5, 2.5, reverse_spacing=True)[:-1],
#     np.sinspace(2.5, 15)
# ])
alphas_nf = np.linspace(alphas_xfoil.min(), alphas_xfoil.max(), 1000)
Re_values_to_test = [1e4, 5e4, 9e4, 2e5, 1e6, 1e7, 1e8]
# Re_values_to_test = [10e3, 50e3, 200e3, 500e3, 5e6, 1e8]
# Re_values_to_test = [1e4, 1e8]

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots(figsize=(8,5))
plt.xscale('log')

cmap = LinearSegmentedColormap.from_list(
    "custom_cmap",
    colors=[
        p.adjust_lightness(c, 0.8) for c in
        ["orange", "darkseagreen", "dodgerblue"]
    ]
)

colors = cmap(np.linspace(0, 1, len(Re_values_to_test)))

transparency = 0.7

for Re, color in tqdm(zip(Re_values_to_test, colors)):

    nf_aero_xl = get_aero_from_airfoil(
        airfoil=af,
        alpha=alphas_nf,
        Re=Re,
        model_size="xxxlarge"
    )

    plt.plot(
        nf_aero_xl["CD"],
        nf_aero_xl["CL"],
        "--",
        color=color, alpha=transparency
    )

    nf_aero_m = get_aero_from_airfoil(
        airfoil=af,
        alpha=alphas_nf,
        Re=Re,
        model_size="medium"
    )

    plt.plot(
        nf_aero_m["CD"],
        nf_aero_m["CL"],
        ":",
        color=color, alpha=transparency
    )

    xfoil_aero = asb.XFoil(
        airfoil=af,
        Re=Re,
        timeout=120,
        max_iter=100,
    ).alpha(alpha=alphas_xfoil)

    plt.plot(
        xfoil_aero["CD"],
        xfoil_aero["CL"],
        color=color, alpha=transparency
    )
    # xfoil_aero = nf_aero

    annotate_x = np.max(np.array([
        nf_aero_xl["CD"][-1],
        nf_aero_m["CD"][-1],
        xfoil_aero["CD"][-1]
    ]))
    annotate_y = np.median(np.array([
        nf_aero_xl["CL"][-1],
        nf_aero_m["CL"][-1],
        xfoil_aero["CL"][-1]
    ]))

    plt.annotate(
        f" $Re = \\mathrm{{{eng_string(Re)}}}$",
        xy=(annotate_x, annotate_y),
        color=p.adjust_lightness(color, 0.8),
        ha="left", va="center", fontsize=10
    )

plt.annotate(
    text="Note the log-scale on $C_D$, which is unconventional - it's\nthe only way to keep it readable given the wide range.",
    xy=(0.01, 0.01),
    xycoords="figure fraction",
    ha="left",
    va="bottom",
    fontsize=8,
    alpha=0.7
)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.legend(
    handles=[
        Line2D([], [], color="k", label="XFoil"),
        Line2D([], [], color="k", linestyle="--", label="NeuralFoil \"xxxlarge\""),
        Line2D([], [], color="k", linestyle=":", label="NeuralFoil \"medium\""),
    ],
    title="Analysis Method",
    # loc=(0.8, 0.15),
    loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5),
    fontsize=11,
    labelspacing=0.3, columnspacing=1.5, handletextpad=0.4,
    framealpha=0.5,
)

plt.xlim()
plt.ylim(bottom=-0.8)

afax = ax.inset_axes([0.76, 0.802, 0.23, 0.23])
afax.fill(
    af.x(), af.y(),
    facecolor=(0, 0, 0, 0.2), linewidth=1, edgecolor=(0, 0, 0, 0.7)
)
afax.annotate(
    text=f"{af.name} Airfoil",
    xy=(0.5, 0.15),
    xycoords="data",
    ha="center",
    va="bottom",
    fontsize=10,
    alpha=0.7
)

afax.grid(False)
afax.set_xticks([])
afax.set_yticks([])
# afax.axis('off')
afax.set_facecolor((1, 1, 1, 0.5))
afax.set_xlim(-0.05, 1.05)
afax.set_ylim(-0.05, 0.28)
afax.set_aspect("equal", adjustable='box')


plt.suptitle("Comparison of $C_L$-$C_D$ Polar for NeuralFoil vs. XFoil", fontsize=16,y=0.94)
plt.title(f"On {af.name} Airfoil (out-of-sample)", fontsize=12, alpha=0.7)

p.show_plot(
    None,
    "Drag Coefficient $C_D$",
    "Lift Coefficient $C_L$",
    legend=False,
    savefig="neuralfoil_point_validation.svg",
    savefig_transparent=False,
    rotate_axis_labels=False
)
