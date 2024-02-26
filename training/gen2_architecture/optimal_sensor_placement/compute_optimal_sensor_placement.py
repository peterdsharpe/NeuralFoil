from load_data import df, Data
import numpy as np

Cfs = np.stack([
    df[f"{side}_bl_H_{i}"]
    for i in range(Data.N)
    for side in [
        "upper",
        # "lower"
    ]
], axis=1)

# Remove rows with any nans
Cfs = Cfs[~np.any(np.isnan(Cfs), axis=1)]

import pysensors as ps

model = ps.SSPOR(n_sensors=30)
model.fit(
    x=Cfs[:300],
)

sensor_ids = np.sort(model.get_selected_sensors())

print(sensor_ids)

import aerosandbox as asb
import aerosandbox.numpy as np

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

af = asb.KulfanAirfoil("naca0012")
af.draw(show=False)
for i in sensor_ids:
    if i < Data.N:
        x = np.linspace(0, 1, Data.N)[i]
        plt.plot(*af.upper_coordinates(x).T, ".k", markersize=10)
    else:
        x = np.linspace(0, 1, Data.N)[i - Data.N]
        plt.plot(*af.lower_coordinates(x).T, ".k", markersize=10)
p.show_plot("Optimal Sensor Locations")
