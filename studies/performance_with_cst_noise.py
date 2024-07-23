import aerosandbox as asb
import aerosandbox.numpy as np
import neuralfoil as nf

af = asb.KulfanAirfoil("naca0012")

typical_weight_magnitude = np.median(
    np.concatenate([np.abs(af.lower_weights), np.abs(af.upper_weights)])
)  # 0.1487

wiggle_basis_vector = (-1) ** np.arange(len(af.lower_weights))  # [1, -1, 1, -1, ...]


@np.vectorize
def get_airfoil_with_wiggly_noise(relative_noise: float) -> asb.KulfanAirfoil:
    noise = relative_noise * typical_weight_magnitude
    return asb.KulfanAirfoil(
        name=f"{af.name} + Wiggly Noise of {relative_noise * .2:%}",
        lower_weights=af.lower_weights + noise * wiggle_basis_vector,
        upper_weights=af.upper_weights + noise * wiggle_basis_vector,
        leading_edge_weight=af.leading_edge_weight,
        TE_thickness=af.TE_thickness,
    )


relative_noises = np.linspace(0, 1, 21)
wiggly_airfoils = get_airfoil_with_wiggly_noise(relative_noises)
nf_aeros = [waf.get_aero_from_neuralfoil(alpha=5, Re=1e6) for waf in wiggly_airfoils]
xf_aeros = [
    asb.XFoil(
        airfoil=waf,
        Re=1e6,
        mach=0,
    ).alpha(5)
    for waf in wiggly_airfoils
]

### Below is just plotting code

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots(2, 1)

ax[0].plot(relative_noises, [aero["CD"] for aero in nf_aeros], label=f"NeuralFoil")
ax[0].plot(relative_noises, [aero["CD"] for aero in xf_aeros], label=f"XFoil")
ax[0].set_ylabel("$C_D$")
ax[0].legend()

ax[1].plot(
    relative_noises,
    [aero["analysis_confidence"] for aero in nf_aeros],
)
ax[1].set_ylabel("NeuralFoil\nAnalysis\nConfidence")

for a in ax:
    a.xaxis.set_major_formatter(p.mpl.ticker.PercentFormatter(xmax=1))

plt.tight_layout(rect=[0.05, 0.1, 1, 0.93], h_pad=2)

# Draw the airfoils
for waf, noise in tuple(zip(wiggly_airfoils, relative_noises))[::5]:
    # Gets the figure-coordinates of the data point on ax[2]
    x, y = ax[1].transData.transform([noise, 0])
    display_center = np.array(
        [
            ax[1].transData.transform([noise, 0])[0],
            ax[1].transAxes.transform([0, 0])[1] - 120,
        ]
    )
    display_size = np.array(
        [
            fig.transFigure.transform([0.1, 0])[0]
            - fig.transFigure.transform([0, 0])[0],
            fig.transFigure.transform([0, 0.1])[1]
            - fig.transFigure.transform([0, 0])[1],
        ]
    )
    display_lowerleft = display_center - display_size / 2
    display_upperright = display_center + display_size / 2
    fig_lowerleft = fig.transFigure.inverted().transform(display_lowerleft)
    fig_upperright = fig.transFigure.inverted().transform(display_upperright)

    afax = fig.add_axes(
        [
            fig_lowerleft[0],
            fig_lowerleft[1],
            fig_upperright[0] - fig_lowerleft[0],
            fig_upperright[1] - fig_lowerleft[1],
        ],
        zorder=10,
    )
    afax.fill(
        waf.x(),
        waf.y(),
        facecolor=(0, 0, 0, 0.2),
        linewidth=1,
        edgecolor=(0, 0, 0, 0.7),
    )
    afax.grid(False)
    afax.axis("off")

    afax.set_xlim(-0.05, 1.05)
    afax.set_ylim(-0.28, 0.28)
    afax.set_aspect("equal", adjustable="box")

plt.annotate(
    text="\"Relative Noise Magnitude\" is the (scale of the added noise) / (median CST weight magnitude).",
    xy=(0.02, 0.02),
    xycoords="figure fraction",
    ha="left",
    color='gray',
    fontsize=9
)


p.show_plot(
    title="Effect of CST Noise on NeuralFoil Performance",
    xlabel="Relative Noise Magnitude [%]",
    tight_layout=False,
    legend=False,
    show=True,
    savefig="performance_with_cst_noise.svg",
)
