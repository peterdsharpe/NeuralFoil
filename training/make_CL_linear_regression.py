import aerosandbox as asb
import aerosandbox.numpy as np
from data.load_data import df_train, df_test, weights, kulfan_cols, aero_input_cols, aero_output_cols, all_cols


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
        p[f"CL0_high_{i}"] * camber_modes[i]
        for i in range(8)
    ]) + p["CL0_high_LE"] * x["kulfan_LE_weight"]
    CL0_low = sum([
        p[f"CL0_low_{i}"] * camber_modes[i]
        for i in range(8)
    ]) + p["CL0_low_LE"] * x["kulfan_LE_weight"]

    CLa_high = p["CLa_high"] + p["delta_CLa_high_area"] * sum(thickness_modes)

    CLa_low = p["CLa_low"] + p["delta_CLa_low_area"] * sum(thickness_modes)

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
    for k, v in df_train[input_cols].to_dict().items()
}

fit = asb.FittedModel(
    model=model,
    x_data=x_data,
    y_data=df_train["CL"].to_numpy(),
    weights=weights,
    parameter_guesses={
        "CLa_low"            : 4,
        "CLa_high"           : 6.28,
        "log10_Re_switch"    : 5,
        "log10_Re_scale"     : 1,
        "CL0_high_LE"        : 0,
        **{f"CL0_high_{i}": 0 for i in range(8)},
        "CL0_low_LE"         : 0,
        **{f"CL0_low_{i}": 0 for i in range(8)},
        "delta_CLa_high_area": 0,
        "delta_CLa_low_area" : 0,
    },
    parameter_bounds={
        "CLa_low"       : (0, None),
        "CLa_high"      : (0, None),
        "log10_Re_scale": (0, None),
    },
    # verbose=False,
)
y_model = fit(fit.x_data)

print(fit.parameters)
print(fit.goodness_of_fit("mean_absolute_error"))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots(figsize=(6, 0.25 * len(fit.parameters)))

    ax.scatter(
        y=list(fit.parameters.keys())[::-1],
        x=list(fit.parameters.values())[::-1],
        color="black",
        marker="|",
        s=100,
        linewidths=2,
        zorder=10,
    )
    ax.axvline(0, color="white", linewidth=3, zorder=3)

    ax.set(
        xlabel="Coefficient Value",
        ylabel="Coefficient Name",
    )
    p.set_ticks(1, 0.2)
    p.show_plot("Linear Regression: Fit Coefficients")

    fig, ax = plt.subplots()

    af = asb.Airfoil("naca4418").normalize().repanel().normalize()
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters

    kulfan_params = get_kulfan_parameters(af.coordinates, n_weights_per_side=8)

    alpha_plot = np.linspace(-10, 10, 21)
    Res = 1e6 * 2. ** np.arange(-5, 5)
    colors = plt.get_cmap("rainbow")(np.linspace(0, 1, len(Res)))
    colors = [p.adjust_lightness(c, 0.6) for c in colors]

    from tqdm import tqdm
    from aerosandbox.tools.string_formatting import eng_string

    for Re, color in tqdm(zip(Res, colors), desc="Sweeping Reynolds Numbers...", total=len(Res), unit="runs"):
        try:
            xf_aero = asb.XFoil(
                airfoil=af,
                Re=Re,
                xfoil_repanel=False,
                timeout=10,
            ).alpha(alpha_plot, start_at=4)

            plt.plot(
                xf_aero["alpha"],
                xf_aero["CL"],
                "-", color=color, label=eng_string(Re), alpha=0.6
            )
        except FileNotFoundError:
            pass

        fit_aero = fit({
            "alpha"              : alpha_plot,
            "Re"                 : Re,
            **{f"kulfan_lower_{i}": kulfan_params["lower_weights"][i] for i in range(8)},
            **{f"kulfan_upper_{i}": kulfan_params["upper_weights"][i] for i in range(8)},
            "kulfan_TE_thickness": kulfan_params["TE_thickness"],
            "kulfan_LE_weight"   : kulfan_params["leading_edge_weight"]
        })
        plt.plot(
            alpha_plot,
            fit_aero,
            ":", color=color, alpha=0.6
        )
    plt.plot([], [], "-k", label="XFoil")
    plt.plot([], [], ":k", label="Linear Model")
    plt.xlim(alpha_plot.min(), alpha_plot.max())
    plt.ylim(-1.5, 1.5)
    p.set_ticks(2, 1, 0.5, 0.1)
    plt.xlabel("Angle of Attack $\\alpha$ [deg]")
    plt.ylabel("Lift Coefficient $C_L$ [-]")
    plt.title(f"{af.name} Airfoil")
    plt.legend(
        title="Reynolds Numbers", fontsize=10, ncols=2, loc="lower right"
    )

    axaf = ax.inset_axes(bounds=(0.05, 0.7, 0.25, 0.25))
    axaf.plot(
        af.x(), af.y(), "-k", alpha=0.8, zorder=100
    )
    axaf.set_aspect("equal")
    axaf.xaxis.set_ticklabels([])
    axaf.yaxis.set_ticklabels([])

    p.show_plot(legend=False)
