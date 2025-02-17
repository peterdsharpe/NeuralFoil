import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import aerosandbox.tools.webplotdigitizer_reader as wr
from pathlib import Path

af = asb.Airfoil("naca0012")
alpha_xf = np.linspace(0, 45, 35)
alpha_nf = np.linspace(0, 180, 300)
Re = 1.8e6

xf_aero = asb.XFoil(
    airfoil=af,
    Re=Re,
    max_iter=100,
    timeout=60,
).alpha(alpha_xf)

nf_aero = af.get_aero_from_neuralfoil(
    alpha=alpha_nf,
    Re=Re,
    model_size="xxxlarge",
)

naca_data_folder = Path(__file__).parent / "poststall"
naca_lift_data = wr.read_webplotdigitizer_csv(naca_data_folder / "lift.csv")
naca_drag_data = wr.read_webplotdigitizer_csv(naca_data_folder / "drag.csv")
naca_moment_data = wr.read_webplotdigitizer_csv(naca_data_folder / "moment.csv")
naca_data = {
    "CL": naca_lift_data,
    "CD": naca_drag_data,
    "CM": naca_moment_data,
}

fig, ax = plt.subplots(3, 1, figsize=(6.5, 5.5), sharex=True)


def plot(ax, title, field):
    plt.sca(ax)
    plt.ylabel(title)
    plt.plot(
        xf_aero["alpha"],
        xf_aero[field],
        ".", label="XFoil",
        color="k", alpha=0.7,
        zorder=4, markeredgewidth=0,
    )

    plt.plot(
        naca_data[field]["data"][:, 0],
        naca_data[field]["data"][:, 1],
        "x", label="Experiment",
        color="k", alpha=0.7,
        zorder=4, markersize=5,
    )

    plt.plot(
        alpha_nf,
        nf_aero[field],
        label="NeuralFoil",
    )


plot(ax[0], "Lift Coeff. $C_L$", "CL")
plot(ax[1], "Drag Coeff. $C_D$", "CD")
plot(ax[2], "Moment Coeff. $C_M$", "CM")

plt.xlabel("Angle of Attack (°)")

ax[0].legend(
    # loc="upper left",
    # ncols=2
)
ax[1].set_ylim(bottom=0)

p.show_plot(show=False, legend=False)
# for deg in np.linspace(0, 360, 9):
#     # Gets the figure-coordinates of the data point on ax[2]
#     x, y = ax[2].transData.transform(
#         [deg, 0]
#     )
#
#     display_center = np.array([
#         ax[2].transData.transform([deg, 0])[0],
#         ax[2].transAxes.transform([0, 0])[1] - 70
#     ])
#     display_size = np.array([
#         fig.transFigure.transform([0.1, 0])[0] - fig.transFigure.transform([0, 0])[0],
#         fig.transFigure.transform([0, 0.1])[1] - fig.transFigure.transform([0, 0])[1]
#     ])
#     display_lowerleft = display_center - display_size / 2
#     display_upperright = display_center + display_size / 2
#     fig_lowerleft = fig.transFigure.inverted().transform(display_lowerleft)
#     fig_upperright = fig.transFigure.inverted().transform(display_upperright)
#
#     afax = fig.add_axes(
#         [
#             fig_lowerleft[0],
#             fig_lowerleft[1],
#             fig_upperright[0] - fig_lowerleft[0],
#             fig_upperright[1] - fig_lowerleft[1]
#         ],
#         zorder=10
#     )
#     afd = af.rotate(-np.radians(deg), x_center=0.5, y_center=0)
#     afax.fill(
#         afd.x(),
#         afd.y(),
#         facecolor=(0, 0, 0, 0.2),
#         linewidth=1,
#         edgecolor=(0, 0, 0, 0.7)
#     )
#     afax.grid(False)
#     afax.axis("off")
#
#     afax.set_xlim(-0.05, 1.05)
#     afax.set_ylim(-0.5, 0.5)
#     afax.set_aspect("equal", adjustable='box')

for a in ax:
    plt.sca(a)
    p.set_ticks(30, 10)

p.show_plot(
    set_ticks=False,
    legend=False,
    tight_layout=False,
    savefig="post_stall_extrapolation.svg"
)
