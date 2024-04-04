import numpy as np
import numba


@numba.njit
def advance_window(window, n):
    if n >= len(window):
        return np.random.randn(len(window))
    if n == 0:
        return window
    window = np.roll(window, -n)
    window[-n:] = np.random.randn(n)
    return window


@numba.njit
def patience_runtime_sample(patience=50, max_trials=1000):
    window = np.random.randn(patience)
    start_shift = np.argmin(window)
    min = window[start_shift]
    window = advance_window(window, start_shift)

    i = 0
    while True:
        if np.all(window > min) or i >= max_trials:
            return start_shift + patience + i
        else:
            window = advance_window(window, 1)
            min = np.minimum(min, window[-1])
            i += 1


@numba.njit
def patience_runtime_distribution(n_samples=1000, patience=50, max_trials=1000):
    return np.array([
        patience_runtime_sample(patience=patience, max_trials=max_trials)
        for _ in range(n_samples)
    ])


import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots()
d = patience_runtime_distribution(
    n_samples=50000,
    patience=50,
    max_trials=1000,
)
p.sns.histplot(d,
               discrete=True
               )
p.show_plot(
    "Patience runtime distribution",
    "Number of trials",
    "Frequency"
)
