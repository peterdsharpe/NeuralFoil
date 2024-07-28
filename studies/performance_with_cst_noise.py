import aerosandbox as asb
import aerosandbox.numpy as np
import neuralfoil as nf
from tqdm import tqdm

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
        upper_weights=af.upper_weights - noise * wiggle_basis_vector,
        leading_edge_weight=af.leading_edge_weight,
        TE_thickness=af.TE_thickness,
    )


relative_noises = np.linspace(0, 2, 101)
wiggly_airfoils = get_airfoil_with_wiggly_noise(relative_noises)
nf_aeros = [
    waf.get_aero_from_neuralfoil(
        alpha=5,
        Re=1e6,
        model_size="xxxlarge",
    )
    for waf in wiggly_airfoils
]
xf_aeros = [
    asb.XFoil(
        airfoil=waf,
        Re=1e6,
        mach=0,
        timeout=3,
    ).alpha(5)
    for waf in tqdm(wiggly_airfoils, desc="XFoil")
]

### Below is just plotting code

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots(2, 1)

nf_cd = [aero["CD"] for aero in nf_aeros]
xf_cd = [aero["CD"] for aero in xf_aeros]
xf_cd = [float(cd[0]) if len(cd) == 1 else np.nan for cd in xf_cd]

ax[0].plot(relative_noises, nf_cd, label=f'NeuralFoil "xxxlarge"', zorder=5)
ax[0].plot(relative_noises, xf_cd, ".k", label=f"XFoil (ground truth)")
ax[0].set_ylabel("$C_D$")
ax[0].legend()

ax[1].plot(
    relative_noises,
    [aero["analysis_confidence"] for aero in nf_aeros],
)
ax[1].set_ylabel("NeuralFoil\nAnalysis\nConfidence")
plt.annotate(
    text="Nose becomes too sharp;\nphysics assumptions starting to break.",
    xy=(0.72, 0.4),
    xytext=(0.9, 0.7),
    xycoords="data",
    fontsize=8,
    arrowprops={
        "color"     : "k",
        "width"     : 0.25,
        "headwidth" : 4,
        "headlength": 6,
    }
)

plt.annotate(
    text="Airfoil becomes self-intersecting;\nanalysis confidence\neffectively goes to zero.",
    xy=(1.2, 0.05),
    xytext=(1.3, 0.30),
    xycoords="data",
    fontsize=8,
    arrowprops={
        "color"     : "k",
        "width"     : 0.25,
        "headwidth" : 4,
        "headlength": 6,
    }
)


for a in ax:
    a.xaxis.set_major_formatter(p.mpl.ticker.PercentFormatter(xmax=1))

plt.tight_layout(rect=[0.05, 0.1, 1, 0.93], h_pad=2)

# Draw the airfoils
draw_indices = np.round(np.linspace(0, len(wiggly_airfoils) - 1, 8)).astype(int)

for waf, noise in tuple(zip(wiggly_airfoils[draw_indices], relative_noises[draw_indices])):
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
    waf = waf.rotate(np.radians(-5))
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
    text='"Relative Noise Magnitude" is the (scale of the added noise) / (median CST weight magnitude).',
    xy=(0.02, 0.02),
    xycoords="figure fraction",
    ha="left",
    color="gray",
    fontsize=9,
)


p.show_plot(
    title="Effect of CST Noise on NeuralFoil Performance",
    xlabel="Relative Noise Magnitude [%]",
    tight_layout=False,
    legend=False,
    show=True,
    savefig="performance_with_cst_noise.svg",
)
