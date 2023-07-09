import aerosandbox as asb
import aerosandbox.numpy as np
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig = plt.figure(figsize=(6, 3), dpi=600)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis('off')

airfoil = asb.Airfoil("dae11").repanel().normalize()

### AF non-discretized
loc = (-2, 0)

plt.plot(
    airfoil.x() + loc[0],
    airfoil.y() + loc[1],
    color=p.adjust_lightness("tomato", 0.4),
    linewidth=1
)

plt.fill(
    airfoil.x() + loc[0],
    airfoil.y() + loc[1],
    color=p.adjust"tomato",
    alpha=0.2,
)

plt.gca().set_aspect("equal", adjustable='box')
p.show_plot()

nn_color = "darkviolet"
