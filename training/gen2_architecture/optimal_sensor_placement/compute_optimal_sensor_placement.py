from load_data import df, Data
import numpy as np

df = df.to_pandas().dropna()

data = np.stack([
    # (df[f"{side}_bl_theta_{i}"])
    (df[f"{side}_bl_H_{i}"])
    for side in [
        "upper",
        "lower",
    ]
    for i in range(Data.N)
], axis=1)

data_flipped = np.stack([
    # (df[f"{side}_bl_theta_{i}"])
    (df[f"{side}_bl_H_{i}"])
    for side in [
        "lower",
        "upper",
    ]
    for i in range(Data.N)
], axis=1)

# Flip and augment
data = np.concatenate([
    data,
    data_flipped
], axis=0)

import pysensors as ps

model = ps.SSPOR(n_sensors=64)
model.fit(
    x=data,
)

sensor_ids = np.sort(model.get_selected_sensors())

print(sensor_ids)

import aerosandbox as asb
import aerosandbox.numpy as np

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots(figsize=(6, 3))

af = asb.KulfanAirfoil("naca0012")
af.draw(show=False)
for i in sensor_ids:
    if i < Data.N:
        x = np.linspace(0, 1, Data.N)[i]
        plt.plot(*af.upper_coordinates(x).T, ".k", markersize=10)
    else:
        x = np.linspace(0, 1, Data.N)[i - Data.N]
        plt.plot(*af.lower_coordinates(x).T, ".k", markersize=10)

subtitle = "Question: \"If I have 64 sensors, where should I put them on a generic airfoil to reconstruct $H(s)$ with minimum error?\""
import textwrap

plt.text( # Top center of axes, textwrapped
    0.5, 0.95,
    textwrap.fill(subtitle, width=65),
    horizontalalignment='center',
    verticalalignment='top',
    transform=plt.gca().transAxes,
    fontsize=11, alpha=0.7,
)

answer = "Answer: \"Roughly evenly across the airfoil.\""

plt.text(
    0.5, 0.05,
    textwrap.fill(answer, width=65),
    horizontalalignment='center',
    verticalalignment='bottom',
    transform=plt.gca().transAxes,
    fontsize=11, alpha=0.7,
)

p.show_plot(
    "Optimal Sensor Locations for Compressed\nSensing of Shape Factor"
)
