import aerosandbox as asb
import aerosandbox.numpy as np
from tqdm import tqdm
import os

n_weights_per_side = 10
datafile = f"{os.getpid()}.csv"
print(datafile)

if not os.path.exists(datafile):
    with open(datafile, "w+") as f:
        f.write(
            ",".join(
                ["alpha"] +
                ["Re"] +
                ["CL", "CD", "CM", "Cpmin", "Top_Xtr", "Bot_Xtr"] +
                [f"kulfan_lower_{i}" for i in range(n_weights_per_side)] +
                [f"kulfan_upper_{i}" for i in range(n_weights_per_side)] +
                [f"kulfan_TE_thickness"] +
                [f"kulfan_LE_weight"]
            )
        )

airfoil_database_path = asb._asb_root / "geometry" / "airfoil" / "airfoil_database"

UIUC_airfoils = [
    asb.Airfoil(name=filename.stem).normalize()
    for filename in airfoil_database_path.iterdir() if filename.suffix == ".dat"
]

from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters, get_kulfan_coordinates

def float_to_str(f: float) -> str:
    if 0 < abs(f) < 1:
        s = f"{f:.8f}"
        s = s.lstrip("0")
    else:
        s = f"{f:.7g}"

    if "." in s:
        s = s.rstrip("0")

    return s

for _ in tqdm(range(1000000)):
    af_1: asb.Airfoil = np.random.choice(UIUC_airfoils)
    af_2: asb.Airfoil = np.random.choice(UIUC_airfoils)


    af = af_1.blend_with_another_airfoil(af_2, blend_fraction=np.random.rand()).normalize()

    assert af.as_shapely_polygon().is_valid

    # alphas = np.linspace(-8, 10, 5) + np.random.randn()
    # Re = float(10 ** (5.5 + np.random.randn()))
    #
    # xf = asb.XFoil(
    #     airfoil=af,
    #     Re=Re,
    #     mach=0,
    #     timeout=5,
    # )
    #
    # try:
    #     aero = xf.alpha(alphas)
    #     # print(aero)
    # except FileNotFoundError:
    #     continue
    #
    # kulfan_params = get_kulfan_parameters(
    #     coordinates=af.coordinates
    # )
    # # print(kulfan_params)
    # # af.draw()
    #
    # with open(datafile, "a+") as f:
    #     for i, alpha in enumerate(aero['alpha']):
    #         numbers = (
    #             [alpha] +
    #             [Re] +
    #             [aero[key][i] for key in ("CL", "CD", "CM", "Cpmin", "Top_Xtr", "Bot_Xtr")] +
    #             list(kulfan_params["lower_weights"]) +
    #             list(kulfan_params["upper_weights"]) +
    #             [kulfan_params["TE_thickness"]] +
    #             [kulfan_params["leading_edge_weight"]]
    #         )
    #
    #         f.write(
    #             ",".join([
    #                 float_to_str(n) for n in numbers
    #             ]) + "\n"
    #         )


# if __name__ == '__main__':

    # af = asb.Airfoil("goe187")
