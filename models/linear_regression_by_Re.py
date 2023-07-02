import aerosandbox as asb
import aerosandbox.numpy as np
from data.load_data import df, kulfan_cols, aero_input_cols, aero_output_cols, all_cols

N = 500000

indices = np.random.choice(len(df), size=N, replace=False)

df = df[indices]


def model(x, p):
    log10_Re = np.log10(x["Re"])

    switch = (log10_Re - p["log10_Re_switch"]) / p["log10_Re_scale"]

    thickness_modes = [
        x[f"kulfan_upper_{i}"] - x[f"kulfan_lower_{i}"]
        for i in range(8)
    ]
    camber_modes = [
        x[f"kulfan_upper_{i}"] + x[f"kulfan_lower_{i}"]
        for i in range(8)
    ]

    CL0_high = sum([
        p[f"delta_CL0_high_{i}"] * camber_modes[i]
        for i in range(8)
    ]) + p["delta_CL0_high_LE"] * x["kulfan_LE_weight"]
    CL0_low = sum([
        p[f"delta_CL0_low_{i}"] * camber_modes[i]
        for i in range(8)
    ]) + p["delta_CL0_low_LE"] * x["kulfan_LE_weight"]

    CLa_high = p["CLa_high"] + sum([
        p[f"delta_CLa_high_{i}"] * thickness_modes[i]
        for i in range(8)
    ])

    CLa_low = p["CLa_low"] + sum([
        p[f"delta_CLa_low_{i}"] * thickness_modes[i]
        for i in range(8)
    ])

    CL0 = np.blend(
        switch,
        CL0_high,
        CL0_low
    )
    CLa = np.blend(
        switch,
        CLa_high,
        CLa_low
    )

    return CL0 + CLa * np.radians(x["alpha"])


input_cols = aero_input_cols + kulfan_cols

x_data = {
    k: v.to_numpy()
    for k, v in df[input_cols].to_dict().items()
}

fit = asb.FittedModel(
    model=model,
    x_data=x_data,
    y_data=df["CL"].to_numpy(),
    parameter_guesses={
        "CLa_low"        : 3.14,
        "CLa_high"       : 6.28,
        "log10_Re_switch": 5,
        "log10_Re_scale" : 1,
        **{f"delta_CL0_high_{i}": 0 for i in range(8)},
        **{f"delta_CL0_low_{i}": 0 for i in range(8)},
        **{"delta_CL0_high_LE": 0},
        **{"delta_CL0_low_LE": 0},
        **{f"delta_CLa_high_{i}": 0 for i in range(8)},
        **{f"delta_CLa_low_{i}": 0 for i in range(8)},
    },
    parameter_bounds={
        "CLa_low"       : (0, None),
        "CLa_high"      : (0, None),
        "log10_Re_scale": (0, None),
    },
    verbose=False,
)
y_model = fit(fit.x_data)

print(fit.parameters)
print(fit.goodness_of_fit("mean_absolute_error"))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p


    fig, ax = plt.subplots(figsize=(6, 10))

    # A lollipop plot of the key-value pairs in `fit.parameters`. Lollipops are horizontal (e.g., barh-ish).
    ax.scatter(
        y=list(fit.parameters.keys())[::-1],
        x=list(fit.parameters.values())[::-1],
        color="black",
        marker="|",
        s=100,
        linewidths=3,
        zorder=10,
    )
    ax.axvline(0, color="black", linestyle="--", zorder=0)

    ax.set(
        xlabel="Coefficient Value",
        ylabel="Coefficient Name",
    )
    # ax.grid(True, alpha=0.2)
    p.set_ticks(1, 0.1)
    p.show_plot()