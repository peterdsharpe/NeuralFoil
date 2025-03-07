import ray
import csv
import aerosandbox as asb
import aerosandbox.numpy as np
from typing import List
import time
from neuralfoil._basic_data_type import Data

print("Initializing Ray...")
ray.init(
    # address="local",
    # _temp_dir="/home/gridsan/pds/tmp/",
    # num_cpus=2,
)

datafile = "data_xfoil.csv"
n_procs = int(ray.cluster_resources()["CPU"])
print(f"Running on {n_procs} processes.")

airfoil_database_path = asb._asb_root / "geometry" / "airfoil" / "airfoil_database"

airfoil_database = [
    asb.Airfoil(name=filename.stem).normalize().to_kulfan_airfoil()
    for filename in airfoil_database_path.glob("*.dat")
]

### Compute the covariance matrix of airfoil shape parameters, for better data generation later
kulfans_database = np.stack(
    [
        np.concatenate(
            [
                airfoil.upper_weights,
                airfoil.lower_weights,
                np.atleast_1d(airfoil.leading_edge_weight),
                np.atleast_1d(airfoil.TE_thickness),
            ]
        )
        for airfoil in airfoil_database
    ],
    axis=0,
)
mean_database = np.mean(kulfans_database, axis=0)
cov_database = np.cov(kulfans_database, rowvar=False)


@ray.remote
class CSVActor:
    def __init__(self, filename):
        self.filename = filename

    @staticmethod
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

    def append_row(self, row: List[float]):

        row = [self.float_to_str(item) for item in row]

        with open(self.filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(row)


@ray.remote
def worker(csv_actor):

    while True:

        n_airfoils_to_combine = 3

        slices = np.random.rand(n_airfoils_to_combine - 1)
        slices = np.sort(slices)
        slices = np.concatenate([[0], slices, [1]])
        weights = np.diff(slices)  # result is N random numbers in [0, 1] that sum to 1

        parent_airfoils = np.random.choice(
            airfoil_database, size=n_airfoils_to_combine, replace=True
        )

        af = asb.KulfanAirfoil(
            name="Reconstructed Airfoil",
            upper_weights=np.dot(
                weights,
                [parent_airfoil.upper_weights for parent_airfoil in parent_airfoils],
                manual=True,
            ),
            lower_weights=np.dot(
                weights,
                [parent_airfoil.lower_weights for parent_airfoil in parent_airfoils],
                manual=True,
            ),
            leading_edge_weight=np.dot(
                weights,
                [
                    parent_airfoil.leading_edge_weight
                    for parent_airfoil in parent_airfoils
                ],
                manual=True,
            ),
            TE_thickness=np.dot(
                weights,
                [parent_airfoil.TE_thickness for parent_airfoil in parent_airfoils],
                manual=True,
            ),
        )

        af = af.scale(1, np.random.lognormal(0, 0.25))

        deviations = np.random.multivariate_normal(
            np.zeros_like(
                mean_database
            ),  # Not including, since we already have a linear combo of 3 airfoils
            cov_database,
        )

        # deviance = np.random.exponential(0.05)
        af.upper_weights += deviations[:8]
        af.lower_weights += deviations[8:16]
        af.leading_edge_weight += deviations[16]
        af.TE_thickness += deviations[17]

        # if not af.as_shapely_polygon().is_valid:
        #     continue

        alphas = (
            np.linspace(-15, 15, 7)
            + np.random.uniform(-2.5, 2.5)
            + 2.5 * np.random.randn()
        )
        Re = float(10 ** (5.5 + 1.5 * np.random.randn()))

        n_crit = np.random.uniform(0, 18)
        if np.random.rand() < 0.8:
            xtr_upper = 1
        else:
            xtr_upper = np.random.uniform(0, 1)
        if np.random.rand() < 0.8:
            xtr_lower = 1
        else:
            xtr_lower = np.random.uniform(0, 1)

        datas = Data.from_xfoil(
            airfoil=af,
            alphas=alphas,
            Re=Re,
            mach=0,
            n_crit=n_crit,
            xtr_upper=xtr_upper,
            xtr_lower=xtr_lower,
            timeout=30,
            max_iter=200,
            # xfoil_command="~/NeuralFoil/training/training_data/xfoil_supercloud"
        )

        for data in datas:
            ray.get(csv_actor.append_row.remote(data.to_vector()))


csv_actor = CSVActor.remote(filename=datafile)

# Start 8 workers
for _ in range(n_procs):
    worker.remote(csv_actor)

# Keep the main thread alive (otherwise the script would end immediately)
while True:
    time.sleep(1)
