import aerosandbox as asb
import aerosandbox.numpy as np
import neuralfoil as nf
import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

# Compute thickness and camber modes, using a generic cambered NACA airfoil
af_base = asb.KulfanAirfoil("naca4412")
thickness_mode = (af_base.upper_weights - af_base.lower_weights) / 2
thickness_mode /= 0.12

camber_mode = (af_base.upper_weights + af_base.lower_weights) / 2
camber_mode /= 0.04
le_camber_mode = af_base.leading_edge_weight / 0.04


def make_airfoil(max_thickness, max_camber):
    return asb.KulfanAirfoil(
        lower_weights=camber_mode * max_camber - thickness_mode * max_thickness,
        upper_weights=camber_mode * max_camber + thickness_mode * max_thickness,
        leading_edge_weight=le_camber_mode * max_camber
    )


### Make a contour plot
# model_size = "xxsmall"

for model_size in ["xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge"]:

    t, c = (
        np.linspace(-0, 1, 200) * 0.3,
        np.linspace(0, 1, 200) * 0.3
    )
    T, C = np.meshgrid(t, c)
    t_f, c_f = T.flatten(), C.flatten()

    aeros = nf.get_aero_from_kulfan_parameters(
        kulfan_parameters=dict(
            lower_weights=(
                    camber_mode.reshape(-1, 1) * c_f.reshape(1, -1)
                    - thickness_mode.reshape(-1, 1) * t_f.reshape(1, -1)),
            upper_weights=(
                    camber_mode.reshape(-1, 1) * c_f.reshape(1, -1)
                    + thickness_mode.reshape(-1, 1) * t_f.reshape(1, -1)),
            leading_edge_weight=le_camber_mode * c_f,
            TE_thickness=0,
        ),
        alpha=5,
        Re=1e6,
        model_size=model_size
    )

    fig, ax = plt.subplots()
    inv_sigmoid = lambda x: np.log(x / (1 - x))

    p.contour(
        T, C,
        aeros["analysis_confidence"].reshape(T.shape),
        # np.log(aeros["CD"]).reshape(T.shape),
        levels=100,
        linelabels=False,
        linelabels_format=lambda x: f"{x:.2f}",
        contour_kwargs=dict(
            alpha=0.2,
        ),
        colorbar_label="Analysis Confidence",
        cmap="viridis"
    )
    plt.clim(0.75, 1)
    p.show_plot(
        f"Model Size: {model_size}",
        r"Thickness $t/c$",
        r"Maximum Camber $\mathrm{max}(y_{\rm mcl} / c)$",
    )
