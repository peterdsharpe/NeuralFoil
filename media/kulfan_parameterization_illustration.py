import aerosandbox as asb
import aerosandbox.numpy as np

af = asb.Airfoil("dae11")

N = 8

from aerosandbox.geometry.airfoil.airfoil_families import (
    get_kulfan_parameters, get_kulfan_coordinates
)

kulfan_parameters = get_kulfan_parameters(
    af.coordinates,
    n_weights_per_side=N,
)

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots(2, 1, figsize=(5.3,5))

# cmap = plt.get_cmap('turbo')

from matplotlib.colors import LinearSegmentedColormap
colors = ["orange", "darkseagreen", "dodgerblue"]#[::-1]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

cmap_samples_nondim = np.linspace(
    0, 1, 2 * N + 1
)
upper_colors = cmap(cmap_samples_nondim[:N])
neutral_color = cmap(cmap_samples_nondim[N])
lower_colors = cmap(cmap_samples_nondim[N + 1:])

alpha = 0.7

for a in ax:
    a.plot(af.x(), af.y(), "-k", zorder=4, alpha=1)
    a.fill(af.x(), af.y(), "-k", zorder=2, alpha=0.2)

for i in range(N):
    ax[0].plot(
        *get_kulfan_coordinates(
            lower_weights=kulfan_parameters["lower_weights"],
            upper_weights=kulfan_parameters["upper_weights"] + np.eye(N)[N - i - 1] * 1,
            leading_edge_weight=kulfan_parameters["leading_edge_weight"],
            TE_thickness=kulfan_parameters["TE_thickness"],
        ).T,
        color=upper_colors[i], alpha=alpha,
    )

for i in range(N):
    ax[0].plot(
        *get_kulfan_coordinates(
            lower_weights=kulfan_parameters["lower_weights"] + np.eye(N)[i] * -1,
            upper_weights=kulfan_parameters["upper_weights"],
            leading_edge_weight=kulfan_parameters["leading_edge_weight"],
            TE_thickness=kulfan_parameters["TE_thickness"],
        ).T,
        color=lower_colors[i], alpha=alpha,
    )


ax[1].plot(
    *get_kulfan_coordinates(
        lower_weights=kulfan_parameters["lower_weights"],
        upper_weights=kulfan_parameters["upper_weights"],
        leading_edge_weight=kulfan_parameters["leading_edge_weight"] + 1,
        TE_thickness=kulfan_parameters["TE_thickness"],
    ).T,
    color=cmap(1.), alpha=alpha,
)

ax[1].plot(
    *get_kulfan_coordinates(
        lower_weights=kulfan_parameters["lower_weights"],
        upper_weights=kulfan_parameters["upper_weights"],
        leading_edge_weight=kulfan_parameters["leading_edge_weight"],
        TE_thickness=kulfan_parameters["TE_thickness"] + 0.2,
    ).T,
    color=cmap(0.), alpha=alpha,
)

for a in ax:
    plt.sca(a)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.18, 0.26)
    p.equal()

ax[0].set_title("Top- and Bottom-Surface Modes (16)")
ax[1].set_title("Leading-Edge-Modification (LEM) and Trailing Edge (TE) Modes")

p.show_plot(
    "NeuralFoil Airfoil Shape Parameterization\n(18 parameters)",
    show=False
)
plt.savefig("kulfan_parameterization_illustration.png", dpi=600)
plt.savefig("kulfan_parameterization_illustration.pdf")
plt.savefig("kulfan_parameterization_illustration.svg")
plt.show()

