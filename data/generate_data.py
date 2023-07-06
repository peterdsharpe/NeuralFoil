import ray
import csv
import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters, get_kulfan_coordinates
from typing import List
import time
import os

ray.init()

datafile = "data.csv"
n_procs = int(ray.cluster_resources()["CPU"])
print(f"Running on {n_procs} processes.")
n_weights_per_side = 8

airfoil_database_path = asb._asb_root / "geometry" / "airfoil" / "airfoil_database"

UIUC_airfoils = [
    asb.Airfoil(name=filename.stem).normalize()
    for filename in airfoil_database_path.iterdir() if filename.suffix == ".dat"
]

if not os.path.exists(datafile):
    with open(datafile, 'w+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["alpha"] +
            ["Re"] +
            ["CL", "CD", "CM", "Cpmin", "Top_Xtr", "Bot_Xtr"] +
            [f"kulfan_lower_{i}" for i in range(n_weights_per_side)] +
            [f"kulfan_upper_{i}" for i in range(n_weights_per_side)] +
            [f"kulfan_TE_thickness"] +
            [f"kulfan_LE_weight"]
        )


@ray.remote
class CSVActor:
    def __init__(self, filename):
        self.filename = filename

    @staticmethod
    def float_to_str(f: float) -> str:
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

    def append_row(self, row: List[float]):

        row = [self.float_to_str(item) for item in row]

        with open(self.filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)


@ray.remote
def worker(csv_actor):

    while True:

        # # Simulate function f() with random sleep
        # time.sleep(random())
        #
        # # This represents the output from your function `f()`.
        # result = [randint(1, 100) / 10.0 for _ in range(20)]
        #
        # # Send the result to the actor for writing to the CSV
        # ray.get(csv_actor.append_row.remote(result))

        rand_airfoil = lambda: np.random.choice(UIUC_airfoils)

        af = rand_airfoil().blend_with_another_airfoil(
            rand_airfoil(),
            blend_fraction=np.random.rand()
        ).blend_with_another_airfoil(
            rand_airfoil(),
            blend_fraction=np.random.rand()
        )

        kulfan_params = get_kulfan_parameters(
            coordinates=af.normalize().coordinates,
            n_weights_per_side=n_weights_per_side,
        )

        deviance = np.random.exponential(0.05)
        kulfan_params["lower_weights"] += np.random.randn(n_weights_per_side) * deviance
        kulfan_params["upper_weights"] += np.random.randn(n_weights_per_side) * deviance
        kulfan_params["leading_edge_weight"] += np.random.randn() * deviance
        kulfan_params["TE_thickness"] += np.random.exponential(0.005)

        if np.random.rand() < 0.5:
            kulfan_params["TE_thickness"] = 0

        if np.random.rand() < 0.75:
            if np.random.rand() < 0.5:
                kulfan_params["lower_weights"][0] = -kulfan_params["upper_weights"][0]
            else:
                kulfan_params["upper_weights"][0] = -kulfan_params["lower_weights"][0]

        af = asb.Airfoil(
            name="Reconstructed Airfoil",
            coordinates=get_kulfan_coordinates(
                **kulfan_params
            )
        )

        if not af.as_shapely_polygon().is_valid:
            continue

        alphas = np.linspace(-10, 10, 5) + np.random.uniform(-2.5, 2.5) + np.random.randn()
        Re = float(10 ** (5.5 + np.random.randn()))
        xf = asb.XFoil(
            airfoil=af,
            Re=Re,
            mach=0,
            timeout=5,
        )
        try:
            aero = xf.alpha(alphas)
        except FileNotFoundError:
            continue

        for i, alpha in enumerate(aero['alpha']):
            numbers = (
                    [alpha] +
                    [Re] +
                    [aero[key][i] for key in ("CL", "CD", "CM", "Cpmin", "Top_Xtr", "Bot_Xtr")] +
                    list(kulfan_params["lower_weights"]) +
                    list(kulfan_params["upper_weights"]) +
                    [kulfan_params["TE_thickness"]] +
                    [kulfan_params["leading_edge_weight"]]
            )
            ray.get(csv_actor.append_row.remote(numbers))


csv_actor = CSVActor.remote(filename=datafile)

# Start 8 workers
for _ in range(n_procs):
    worker.remote(csv_actor)

# Keep the main thread alive (otherwise the script would end immediately)
while True:
    time.sleep(1)
