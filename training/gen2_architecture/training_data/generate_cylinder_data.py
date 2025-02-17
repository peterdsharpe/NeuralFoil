import csv
import aerosandbox as asb
import aerosandbox.numpy as np
from typing import List
from neuralfoil.gen2_architecture._basic_data_type import Data
from aerosandbox.library.aerodynamics.viscous import Cd_cylinder

datafile = "data_cylinder.csv"


def float_to_str(f: float) -> str:
    if np.isnan(f):
        return ""  # Polars will read this as a null value

    s = f"{f:.8g}"

    if len(s) > 2 and s[:2] == "0.":
        s = s[1:]

    if "." in s:
        s = s.rstrip("0")

    if s[-1] == ".":
        s = s[:-1]

    if s == "." or s == "" or s == "-0":
        s = "0"

    return s


def append_row(row: List[float]):

    row = [float_to_str(item) for item in row]

    with open(datafile, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)


def make_cylinder(N=320):
    s = np.sort(np.random.rand(N))
    s[0] = 0
    s[-1] = 1

    theta = 2 * np.pi * s

    cylinder = asb.Airfoil(
        name="Cylinder",
        coordinates=np.stack([0.5 + 0.5 * np.cos(theta), 0.5 * np.sin(theta)], axis=1),
    )
    cylinder = cylinder.to_kulfan_airfoil()

    return cylinder


# # Plot parameter distributions as violin plots
# cylinders = [make_cylinder() for _ in range(100)]
# df = pd.DataFrame({
#     **{f"upper_weights_{i}": [c.upper_weights[i] for c in cylinders] for i in range(8)},
#     **{f"lower_weights_{i}": [c.lower_weights[i] for c in cylinders] for i in range(8)},
#     "leading_edge_weight": [c.leading_edge_weight for c in cylinders],
#     "TE_thickness * 100"       : [c.TE_thickness * 100 for c in cylinders],
# })
# p.sns.violinplot(data=df, orient="h", linewidth=1)
# plt.xlim(-10, 10)
# # plt.show()
# p.set_ticks(2, 1)
# p.show_plot("Cylinder Airfoil Parameter Distributions", set_ticks=False)


def make_data():

    cylinder = make_cylinder()
    alpha = np.random.uniform(-180, 180)
    Re = float(10 ** (5.5 + 1.5 * np.random.randn()))
    mach = 0
    n_crit = np.random.uniform(4, 14)
    xtr_upper = 1
    xtr_lower = 1

    return Data(
        airfoil=cylinder,
        alpha=alpha,
        Re=Re,
        mach=mach,
        n_crit=n_crit,
        xtr_upper=xtr_upper,
        xtr_lower=xtr_lower,
        analysis_confidence=0.1,
        af_outputs={
            "CL": 0,
            "CD": Cd_cylinder(
                Re_D=Re,
                mach=mach,
                include_mach_effects=mach != 0,
            ),
            "CM": 0,
            "Top_Xtr": np.nan,
            "Bot_Xtr": np.nan,
        },
    )


while True:
    data = make_data()
    append_row(data.to_vector())
