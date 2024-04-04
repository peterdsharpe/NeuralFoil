import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
from tqdm import tqdm

af = asb.Airfoil("naca0012")
deflection_range = (0, 60)
hinge_point_x = 0.7
Re = 1e6

xf_deflections = np.linspace(*deflection_range, 21)
xf_aeros = [
    asb.XFoil(
        airfoil=af.add_control_surface(
            deflection=d,
            hinge_point_x=hinge_point_x
        ),
        Re=Re,
    ).alpha(0)
    for d in tqdm(xf_deflections)
]
nf_deflections = np.linspace(*deflection_range, 61)
nf_aeros = [
    af.add_control_surface(
        deflection=d,
        hinge_point_x=hinge_point_x
    ).get_aero_from_neuralfoil(
        alpha=0,
        Re=Re,
        model_size="xxxlarge",
    )
    for d in tqdm(nf_deflections)
]

fig, ax = plt.subplots(3, 1, figsize=(6.5, 4.5), sharex=True)


def plot(ax, title, field):
    plt.sca(ax)
    plt.ylabel(title)
    plt.plot(
        xf_deflections,
        [
            a[field][0] if len(a[field]) != 0 else np.nan
         for a in xf_aeros
        ],
        ".",
        color="k",
        zorder=4,
    )
    plt.plot(
        nf_deflections,
        [a[field] for a in nf_aeros],
    )


plot(ax[0], "Lift Coeff. $C_L$", "CL")
plot(ax[1], "Drag Coeff. $C_D$", "CD")
plot(ax[2], "Moment Coeff. $C_M$", "CM")

plt.xlabel("\nControl Surface Deflection (°)")

ax[0].legend(
    ["XFoil", "NeuralFoil"],
    # loc="upper left",
    ncols=2
)
ax[1].set_ylim(bottom=0)

# ax[0].set_title(
#     "NACA0012 with trailing-edge flap at $x/c = 0.7$, $\\mathrm{Re} = 10^6$"
# )
# plt.suptitle(
#     "NeuralFoil Accuracy with Varying Control Surface Deflections"
# )

p.show_plot(show=False)
for deg in [0, 20, 40, 60]:
    # Gets the figure-coordinates of the data point on ax[2]
    x, y = ax[2].transData.transform(
        [deg, 0]
    )

    display_center = np.array([
        ax[2].transData.transform([deg, 0])[0],
        ax[2].transAxes.transform([0, 0])[1] - 70
    ])
    display_size = np.array([
        fig.transFigure.transform([0.1, 0])[0] - fig.transFigure.transform([0, 0])[0],
        fig.transFigure.transform([0, 0.1])[1] - fig.transFigure.transform([0, 0])[1]
    ])
    display_lowerleft = display_center - display_size / 2
    display_upperright = display_center + display_size / 2
    fig_lowerleft = fig.transFigure.inverted().transform(display_lowerleft)
    fig_upperright = fig.transFigure.inverted().transform(display_upperright)

    afax = fig.add_axes(
        [
            fig_lowerleft[0],
            fig_lowerleft[1],
            fig_upperright[0] - fig_lowerleft[0],
            fig_upperright[1] - fig_lowerleft[1]
        ],
        zorder=10
    )
    afd = af.add_control_surface(
        deflection=deg,
        hinge_point_x=hinge_point_x
    )
    afax.fill(
        afd.x(),
        afd.y(),
        facecolor=(0, 0, 0, 0.2),
        linewidth=1,
        edgecolor=(0, 0, 0, 0.7)
    )
    afax.grid(False)
    afax.axis("off")

    afax.set_xlim(-0.05, 1.05)
    afax.set_ylim(-0.28, 0.28)
    afax.set_aspect("equal", adjustable='box')

p.show_plot(
    tight_layout=False,
    savefig="control_surface_accuracy.svg"
)
