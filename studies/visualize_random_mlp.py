import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
import aerosandbox.numpy as np

lims = np.array([-1, 1]) * 100

x, y = (
    np.linspace(*lims, 500),
    np.linspace(*lims, 500)
)
X, Y = np.meshgrid(x, y)

np.random.seed(170)
n_hidden_layers = 5
n_hidden_units = 50

W = [
        np.random.randn(n_hidden_units, 2) / np.sqrt(2)
    ] + [
        np.random.randn(n_hidden_units, n_hidden_units) / np.sqrt(n_hidden_units)
        for _ in range(n_hidden_layers - 1)
    ] + [
        np.random.randn(1, n_hidden_units) / np.sqrt(n_hidden_units)
    ]

b = [
        np.random.randn(n_hidden_units)
        for _ in range(n_hidden_layers)
    ] + [
        np.random.randn(1)
    ]


def net(x):
    for i in range(n_hidden_layers + 1):
        x = W[i] @ x + b[i][:, None]
        print("dense")
        if i < n_hidden_layers:
            # x = np.tanh(x)
            # print("tanh")
            # x = np.softplus(x)
            # print("softplus")
            x = np.swish(x)
            print("swish")
            # x = np.maximum(0, x)
            # print("relu")

    return x


Z = net(np.vstack([X.flatten(), Y.flatten()])).reshape(X.shape)

# fig, ax = p.figure3d()
# ax.plot_surface(X, Y, Z, linewidth=0, cmap="viridis")
# ax.view_init(60, 45)

fig, ax = plt.subplots()
p.contour(
    X, Y, Z,
    levels=50,
    linelabels=False,
)
p.show_plot()
