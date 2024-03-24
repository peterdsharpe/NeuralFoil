import aerosandbox.numpy as np
from scipy import stats


def vector_2d(mag, angle_deg):
    angle = np.radians(angle_deg)
    return mag * np.array([np.cos(angle), np.sin(angle)])


cov_principle_components = np.array([
    vector_2d(3, 45),
    vector_2d(0.3, -45)
]).T
eigenvalues = np.linalg.norm(cov_principle_components, axis=0)
eigenvectors = cov_principle_components / eigenvalues

cov = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)

assert np.allclose(np.linalg.eig(cov).eigenvalues, eigenvalues)
for v1, v2 in zip(np.linalg.eig(cov).eigenvectors.T, eigenvectors.T):
    assert np.allclose(v1, v2) or np.allclose(v1, -v2)
assert np.allclose(eigenvalues * eigenvectors, cov_principle_components)

n = 10000
data = np.random.multivariate_normal(
    mean=[1, 1],
    cov=cov,
    size=n
)

mean = np.mean(data, axis=0)

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p
fig, ax = plt.subplots()
x, y= data.T
p.sns.scatterplot(
    x=x, y=y,
    s=5, color="black", alpha=0.5
)
# p.sns.kdeplot(x=x, y=y, levels=5, color="k", linewidths=1)
ax.set_aspect('equal', 'box')
# plt.show()

# Mahalanobis distance
X, Y = np.meshgrid(
    np.linspace(-5, 7, 100),
    np.linspace(-5, 7, 100)
)
x_f, y_f = X.flatten(), Y.flatten()

point = np.stack([x_f, y_f], axis=1)
# mahalanobis_distance_squared = (point - mean).T @ np.linalg.inv(cov) @ (point - mean).T
mahalanobis_distance_squared = np.sum((point - mean) @ np.linalg.inv(cov) * (point - mean), axis=1)

# fig, ax = plt.subplots()
p.contour(
    X, Y, np.sqrt(mahalanobis_distance_squared.reshape(X.shape)),
    levels=np.arange(10), alpha=0.5
)
p.show_plot()