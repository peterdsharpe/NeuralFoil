import aerosandbox as asb
import aerosandbox.numpy as np
from typing import Union, Dict, Set, List, Iterable
from pathlib import Path
import re
from neuralfoil._basic_data_type import Data

nn_weights_dir = Path(__file__).parent / "nn_weights_and_biases"

# These are the x-coordinates on the top and bottom surfaces of the airfoil where detailed boundary layer data is computed.
# It is made externally-accessible (`nf.bl_x_points`) in case you want to use it dynamically.
bl_x_points = Data.bl_x_points

# Here, we compute a small epsilon value, which is used later to clip values to suppress overflow.
# This looks a bit complicated below, but it's basically just a dynamic way to avoid explicitly referring to float bit-widths.
_zero, _one, _inf = [np.array(x) for x in [0, 1, np.inf]]
_eps: float = (
    np.maximum(np.nextafter(_zero, _one), 1 / np.nextafter(_inf, _zero)) * 10
)  # Adds a bit of padding, to be safe.
_ln_eps: float = np.log(_eps)


def _sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    x = np.clip(x, _ln_eps, -_ln_eps)  # Clip to suppress overflow
    return 1 / (1 + np.exp(-x))


### For speed, pre-loads parameters with statistics about the training distribution
# Includes the mean, covariance, and inverse covariance of training data in the input latent space (25-dim)
_scaled_input_distribution = dict(
    np.load(nn_weights_dir / "scaled_input_distribution.npz")
)
_scaled_input_distribution["N_inputs"]: int = len(
    _scaled_input_distribution["mean_inputs_scaled"]
)

### For speed, pre-loads the neural network weights and biases
_nn_parameter_files: Iterable[Path] = nn_weights_dir.glob("nn-*.npz")
_allowable_model_sizes: set[str] = set(
    [
        # regex parse, which results in the strings "large", "medium", "small", etc.
        re.search(r"nn-(.*).npz", str(path)).group(1)
        for path in _nn_parameter_files
    ]
)
_nn_parameters: dict[str, dict[str, np.ndarray]] = {
    model_size: dict(np.load(nn_weights_dir / f"nn-{model_size}.npz"))
    for model_size in _allowable_model_sizes
}


def _squared_mahalanobis_distance(x: np.ndarray) -> np.ndarray:
    """
    Computes the squared Mahalanobis distance of a set of points from the training data.

    Args:
        x: Query point in the input latent space. Shape: (N_cases, N_inputs)
            For non-vectorized queries, N_cases=1.

    Returns:
        The squared Mahalanobis distance. Shape: (N_cases,)
    """
    d = _scaled_input_distribution
    mean = np.reshape(d["mean_inputs_scaled"], (1, -1))
    x_minus_mean = (x.T - mean.T).T
    return np.sum(x_minus_mean @ d["inv_cov_inputs_scaled"] * x_minus_mean, axis=1)


def get_aero_from_kulfan_parameters(
    kulfan_parameters: dict[str, Union[float, np.ndarray]],
    alpha: Union[float, np.ndarray],
    Re: Union[float, np.ndarray],
    n_crit: Union[float, np.ndarray] = 9.0,
    xtr_upper: Union[float, np.ndarray] = 1.0,
    xtr_lower: Union[float, np.ndarray] = 1.0,
    model_size="large",
) -> dict[str, Union[float, np.ndarray]]:
    """
    Computes aerodynamic coefficients and boundary layer parameters for an aerodynamics case.

    Args:

        kulfan_parameters: The Kulfan (CST) parameters for the airfoil. Should be a dictionary with the
        following keys:

            - "lower_weights": np.ndarray of shape (8,) with the weights for the lower CST coefficients.
            From the leading edge to the trailing edge.

            - "upper_weights": np.ndarray of shape (8,) with the weights for the upper CST coefficients.
            From the leading edge to the trailing edge.

            - "leading_edge_weight": float with the weight for the leading edge thickness.

            - "TE_thickness": float with the trailing edge thickness.

            All can be vectorized by appending a leading dimension to all arrays.

        alpha: Angle of attack in degrees.

        Re: Reynolds number.

        n_crit: Critical amplification factor for natural turbulent transition. Guidelines, from XFoil manual:
            situation                Ncrit
            -----------------        -----
            sailplane                12-14
            motorglider              11-13
            clean wind tunnel        10-12
            average wind tunnel        9     <=  standard "e^9 method"
            dirty wind tunnel         4-8

        xtr_upper: Forced transition location on the upper surface, as a fraction of chord (x/c). 1.0
        allows fully natural transition.

        xtr_lower: Forced transition location on the lower surface, as a fraction of chord (x/c). 1.0
        allows fully natural transition.

        model_size: The size of the neural network to use. Must be one of:
            - "xxsmall"
            - "xsmall"
            - "small"
            - "medium"
            - "large"
            - "xlarge"
            - "xxlarge"
            - "xxxlarge"
            Results in a speed-accuracy tradeoff. The larger the model, the more accurate the results, but the slower
            the computation. The default is "large".

    Returns: A dictionary with the following keys:

        - "analysis_confidence": Confidence of the neural network in its prediction. A value of 1.0 indicates high
        confidence, while a value of 0.0 indicates low confidence.

        - "CL": Lift coefficient.

        - "CD": Drag coefficient.

        - "CM": Moment coefficient.

        - "Top_Xtr": Transition location on the upper surface, as a fraction of chord (x/c).

        - "Bot_Xtr": Transition location on the lower surface, as a fraction of chord (x/c).

        - "upper_bl_theta_i": Angle of attack of the boundary layer at the i-th panel on the upper surface, in degrees.

        - "upper_bl_H_i": Displacement thickness of the boundary layer at the i-th panel on the upper surface.

        - "upper_bl_ue/vinf_i": Ratio of the edge velocity to the freestream velocity at the i-th panel on the upper surface.

        - "lower_bl_theta_i": Angle of attack of the boundary layer at the i-th panel on the lower surface, in degrees.

        - "lower_bl_H_i": Displacement thickness of the boundary layer at the i-th panel on the lower surface.

        - "lower_bl_ue/vinf_i": Ratio of the edge velocity to the freestream velocity at the i-th panel on the lower surface.

        All values are returned as numpy arrays, possibly vectorized if the inputs are vectorized.
    """
    ### Validate inputs
    if model_size not in _allowable_model_sizes:
        raise ValueError(
            f"Invalid {model_size=}. Must be one of {_allowable_model_sizes}."
        )
    nn_params: dict[str, np.ndarray] = _nn_parameters[model_size]

    ### Prepare the inputs for the neural network
    input_rows: List[Union[float, np.ndarray]] = [
        *[kulfan_parameters["upper_weights"][i] for i in range(8)],
        *[kulfan_parameters["lower_weights"][i] for i in range(8)],
        kulfan_parameters["leading_edge_weight"],
        kulfan_parameters["TE_thickness"] * 50,
        np.sind(2 * alpha),
        np.cosd(alpha),
        1 - np.cosd(alpha) ** 2,
        (np.log(Re) - 12.5) / 3.5,
        # No mach
        (n_crit - 9) / 4.5,
        xtr_upper,
        xtr_lower,
    ]

    ### Handle the vectorization, where here we figure out how many cases the user wants to run
    N_cases = 1  # TODO rework this with np.atleast1d
    for row in input_rows:
        if np.length(row) > 1:
            if N_cases == 1:
                N_cases = np.length(row)
            else:
                if np.length(row) != N_cases:
                    raise ValueError(
                        f"The inputs to the neural network must all have the same length. (Conflicting lengths: {N_cases} and {np.length(row)})"
                    )

    for i, row in enumerate(input_rows):
        input_rows[i] = np.ones(N_cases) * row

    x = np.stack(input_rows, axis=1)  # shape: (N_cases, N_inputs)
    ##### Evaluate the neural network

    ### First, determine what the structure of the neural network is (i.e., how many layers it has) by looking at the keys.
    # These keys come from the dictionary of saved weights/biases for the specified neural network.
    try:
        layer_indices: Set[int] = set(
            [int(key.split(".")[1]) for key in nn_params.keys()]
        )
    except TypeError:
        raise ValueError(
            f"Got an unexpected neural network file format.\n"
            f"Dictionary keys should be strings of the form 'net.0.weight', 'net.0.bias', 'net.2.weight', etc.'.\n"
            f"Instead, got keys of the form {nn_params.keys()}.\n"
        )
    layer_indices: List[int] = sorted(list(layer_indices))

    ### Now, set up evaluation of the basic neural network.
    def net(x: np.ndarray) -> np.ndarray:
        """
        Evaluates the raw network (taking in scaled inputs and returning scaled outputs).

        Works in the input and output latent spaces.

        Input `x` shape: (N_cases, N_inputs)
        Output `y` shape: (N_cases, N_outputs)
        """
        x = np.transpose(x)
        layer_indices_to_iterate = layer_indices.copy()

        while len(layer_indices_to_iterate) != 0:
            i = layer_indices_to_iterate.pop(0)
            w = nn_params[f"net.{i}.weight"]
            b = nn_params[f"net.{i}.bias"]
            x = w @ x + np.reshape(b, (-1, 1))

            if (
                len(layer_indices_to_iterate) != 0
            ):  # Don't apply the activation function on the last layer
                x = np.swish(x)
        x = np.transpose(x)
        return x

    y = net(x)  # N_outputs x N_cases
    y[:, 0] = y[:, 0] - _squared_mahalanobis_distance(x) / (
        2 * _scaled_input_distribution["N_inputs"]
    )
    # This was baked into training in order to ensure the network asymptotes to zero analysis confidence far away from the training data.

    ### Then, flip the inputs and evaluate the network again.
    # The goal here is to embed the invariant of "symmetry across alpha" into the network evaluation.
    # (This was also performed during training, so the network is "intended" to be evaluated this way.)

    x_flipped = (
        x + 0.0
    )  # This is an array-api-agnostic way to force a memory copy of the array to be made.
    x_flipped[:, :8] = (
        x[:, 8:16] * -1
    )  # switch kulfan_lower with a flipped kulfan_upper
    x_flipped[:, 8:16] = (
        x[:, :8] * -1
    )  # switch kulfan_upper with a flipped kulfan_lower
    x_flipped[:, 16] = -1 * x[:, 16]  # flip kulfan_LE_weight
    x_flipped[:, 18] = -1 * x[:, 18]  # flip sin(2a)
    x_flipped[:, 23] = x[:, 24]  # flip xtr_upper with xtr_lower
    x_flipped[:, 24] = x[:, 23]  # flip xtr_lower with xtr_upper

    y_flipped = net(x_flipped)
    y_flipped[:, 0] = y_flipped[:, 0] - _squared_mahalanobis_distance(x_flipped) / (
        2 * _scaled_input_distribution["N_inputs"]
    )
    # This was baked into training in order to ensure the network asymptotes to zero analysis confidence far away from the training data.

    ### The resulting outputs will also be flipped, so we need to flip them back to their normal orientation
    y_unflipped = (
        y_flipped + 0.0
    )  # This is an array-api-agnostic way to force a memory copy of the array to be made.
    y_unflipped[:, 1] = y_flipped[:, 1] * -1  # CL
    y_unflipped[:, 3] = y_flipped[:, 3] * -1  # CM
    y_unflipped[:, 4] = y_flipped[:, 5]  # switch Top_Xtr with Bot_Xtr
    y_unflipped[:, 5] = y_flipped[:, 4]  # switch Bot_Xtr with Top_Xtr

    # switch upper and lower Ret, H
    y_unflipped[:, 6 : 6 + 32 * 2] = y_flipped[:, 6 + 32 * 3 : 6 + 32 * 5]
    y_unflipped[:, 6 + 32 * 3 : 6 + 32 * 5] = y_flipped[:, 6 : 6 + 32 * 2]

    # switch upper_bl_ue/vinf with lower_bl_ue/vinf
    y_unflipped[:, 6 + 32 * 2 : 6 + 32 * 3] = -1 * y_flipped[:, 6 + 32 * 5 : 6 + 32 * 6]
    y_unflipped[:, 6 + 32 * 5 : 6 + 32 * 6] = -1 * y_flipped[:, 6 + 32 * 2 : 6 + 32 * 3]

    ### Then, average the two outputs to get the "symmetric" result
    y_fused = (y + y_unflipped) / 2
    y_fused[:, 0] = _sigmoid(y_fused[:, 0])  # Analysis confidence, a binary variable
    y_fused[:, 4] = np.clip(y_fused[:, 4], 0, 1)  # Top_Xtr
    y_fused[:, 5] = np.clip(y_fused[:, 5], 0, 1)  # Bot_Xtr

    ### Unpack outputs
    analysis_confidence = y_fused[:, 0]
    CL = y_fused[:, 1] / 2
    CD = np.exp((y_fused[:, 2] - 2) * 2)
    CM = y_fused[:, 3] / 20
    Top_Xtr = y_fused[:, 4]
    Bot_Xtr = y_fused[:, 5]

    upper_bl_ue_over_vinf = y_fused[:, 6 + Data.N * 2 : 6 + Data.N * 3]
    lower_bl_ue_over_vinf = y_fused[:, 6 + Data.N * 5 : 6 + Data.N * 6]

    upper_theta = ((10 ** y_fused[:, 6 : 6 + Data.N]) - 0.1) / (
        np.abs(upper_bl_ue_over_vinf) * np.reshape(Re, (-1, 1))
    )
    upper_H = 2.6 * np.exp(y_fused[:, 6 + Data.N : 6 + Data.N * 2])

    lower_theta = ((10 ** y_fused[:, 6 + Data.N * 3 : 6 + Data.N * 4]) - 0.1) / (
        np.abs(lower_bl_ue_over_vinf) * np.reshape(Re, (-1, 1))
    )
    lower_H = 2.6 * np.exp(y_fused[:, 6 + Data.N * 4 : 6 + Data.N * 5])

    results = {
        "analysis_confidence": analysis_confidence,
        "CL": CL,
        "CD": CD,
        "CM": CM,
        "Top_Xtr": Top_Xtr,
        "Bot_Xtr": Bot_Xtr,
        **{f"upper_bl_theta_{i}": upper_theta[:, i] for i in range(Data.N)},
        **{f"upper_bl_H_{i}": upper_H[:, i] for i in range(Data.N)},
        **{f"upper_bl_ue/vinf_{i}": upper_bl_ue_over_vinf[:, i] for i in range(Data.N)},
        **{f"lower_bl_theta_{i}": lower_theta[:, i] for i in range(Data.N)},
        **{f"lower_bl_H_{i}": lower_H[:, i] for i in range(Data.N)},
        **{f"lower_bl_ue/vinf_{i}": lower_bl_ue_over_vinf[:, i] for i in range(Data.N)},
    }
    return {key: np.reshape(value, -1) for key, value in results.items()}


def get_aero_from_airfoil(
    airfoil: Union[asb.Airfoil, asb.KulfanAirfoil],
    alpha: Union[float, np.ndarray],
    Re: Union[float, np.ndarray],
    n_crit: Union[float, np.ndarray] = 9.0,
    xtr_upper: Union[float, np.ndarray] = 1.0,
    xtr_lower: Union[float, np.ndarray] = 1.0,
    model_size="large",
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Computes aerodynamic coefficients and boundary layer parameters for an aerodynamics case.

    Geometry is defined by an Airfoil object. Essentially, this function is a higher level of abstraction
    around `get_aero_from_kulfan_parameters()`.

    Args:
        airfoil: An Airfoil or KulfanAirfoil object using AeroSandbox.

        For other arguments, see `get_aero_from_kulfan_parameters()`.

    Returns: A dictionary with outputs. See `get_aero_from_kulfan_parameters()` for details.

    """

    ### Normalize the inputs and evaluate
    normalization_outputs = airfoil.normalize(return_dict=True)
    normalized_airfoil = normalization_outputs["airfoil"].to_kulfan_airfoil(
        n_weights_per_side=8,
        normalize_coordinates=False,  # No need to redo this
    )
    delta_alpha = normalization_outputs["rotation_angle"]  # degrees
    x_translation_LE = normalization_outputs["x_translation"]
    y_translation_LE = normalization_outputs["y_translation"]
    scale = normalization_outputs["scale_factor"]

    x_translation_qc = (
        -x_translation_LE + 0.25 * (1 / scale * np.cosd(delta_alpha)) - 0.25
    )
    y_translation_qc = -y_translation_LE + 0.25 * (1 / scale * np.sind(-delta_alpha))

    raw_aero = get_aero_from_kulfan_parameters(
        kulfan_parameters=normalized_airfoil.kulfan_parameters,
        alpha=alpha + delta_alpha,
        Re=Re / scale,
        n_crit=n_crit,
        xtr_upper=xtr_upper,
        xtr_lower=xtr_lower,
        model_size=model_size,
    )

    ### Correct the force vectors and lift-induced moment from translation
    extra_CM = -raw_aero["CL"] * x_translation_qc + raw_aero["CD"] * y_translation_qc
    raw_aero["CM"] = raw_aero["CM"] + extra_CM

    return raw_aero


def get_aero_from_coordinates(
    coordinates: np.ndarray,
    alpha: Union[float, np.ndarray],
    Re: Union[float, np.ndarray],
    n_crit: Union[float, np.ndarray] = 9.0,
    xtr_upper: Union[float, np.ndarray] = 1.0,
    xtr_lower: Union[float, np.ndarray] = 1.0,
    model_size="large",
):
    """
    Computes aerodynamic coefficients and boundary layer parameters for an aerodynamics case.

    Geometry is defined by a set of (x, y) coordinates. Essentially, this function is a higher level of
    abstraction around `get_aero_from_kulfan_parameters()`.

    Args:

        coordinates: A numpy array of shape (N_points, 2) with the x and y coordinates of the airfoil.
        Should be ordered in standard Selig format (i.e., starting from the upper-surface trailing edge,
        going around the airfoil in a counterclockwise direction, ending at the lower-surface trailing edge).

        For other arguments, see `get_aero_from_kulfan_parameters()`.

    Returns: A dictionary with outputs. See `get_aero_from_kulfan_parameters()` for details.

    """
    return get_aero_from_airfoil(
        airfoil=asb.Airfoil(coordinates=coordinates),
        alpha=alpha,
        Re=Re,
        n_crit=n_crit,
        xtr_upper=xtr_upper,
        xtr_lower=xtr_lower,
        model_size=model_size,
    )


def get_aero_from_dat_file(
    filename,
    alpha: Union[float, np.ndarray],
    Re: Union[float, np.ndarray],
    n_crit: Union[float, np.ndarray] = 9.0,
    xtr_upper: Union[float, np.ndarray] = 1.0,
    xtr_lower: Union[float, np.ndarray] = 1.0,
    model_size="large",
):
    """
    Computes aerodynamic coefficients and boundary layer parameters for an aerodynamics case.

    Geometry is defined by a set of (x, y) coordinates in a .dat file. Essentially, this function is a
    higher level of abstraction around `get_aero_from_kulfan_parameters()`.

    Args:

        filename: A string with the path to the standard Selig-formatted .dat file defining an airfoil shape.

        For other arguments, see `get_aero_from_kulfan_parameters()`.

    Returns: A dictionary with outputs. See `get_aero_from_kulfan_parameters()` for details.

    """
    with open(filename, "r") as f:
        raw_text = f.readlines()

    from aerosandbox.geometry.airfoil.airfoil_families import (
        get_coordinates_from_raw_dat,
    )

    return get_aero_from_coordinates(
        coordinates=get_coordinates_from_raw_dat(raw_text=raw_text),
        alpha=alpha,
        Re=Re,
        n_crit=n_crit,
        xtr_upper=xtr_upper,
        xtr_lower=xtr_lower,
        model_size=model_size,
    )


if __name__ == "__main__":
    airfoil = asb.Airfoil("dae11").repanel().normalize()
    # airfoil = asb.Airfoil("naca0050")
    # airfoil = asb.Airfoil("naca0012").add_control_surface(10, hinge_point_x=0.5)

    alpha = np.linspace(-5, 15, 50)
    Re = 1e6

    aero = get_aero_from_airfoil(airfoil, 3, Re, model_size="xxxlarge")

    aeros = {}

    model_sizes = ["xxxlarge"]

    for model_size in model_sizes:
        aeros[f"NF-{model_size}"] = get_aero_from_airfoil(
            airfoil=airfoil, alpha=alpha, Re=Re, model_size=model_size
        )

    if True:
        aeros["XFoil"] = asb.XFoil(
            airfoil=airfoil, Re=Re, max_iter=20, xfoil_repanel=True
        ).alpha(alpha)

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    for label, aero in aeros.items():
        if "xfoil" in label.lower():
            a = aero["alpha"]
            kwargs = dict(color="k")
        else:
            a = alpha
            kwargs = dict(
                alpha=0.6,
            )

        ax[0, 0].plot(a, aero["CL"], **kwargs)
        ax[0, 1].plot(a, aero["CD"], label=label, **kwargs)
        ax[1, 0].plot(a, aero["CM"], **kwargs)
        ax[1, 1].plot(aero["CD"], aero["CL"], **kwargs)
    plt.sca(ax[0, 1])
    plt.legend()
    ax[0, 0].set_xlabel("$\\alpha$")
    ax[0, 0].set_ylabel("$C_L$")

    ax[0, 1].set_xlabel("$\\alpha$")
    ax[0, 1].set_ylabel("$C_D$")
    ax[0, 1].set_ylim(bottom=0)

    ax[1, 0].set_xlabel("$\\alpha$")
    ax[1, 0].set_ylabel("$C_M$")

    ax[1, 1].set_xlabel("$C_D$")
    ax[1, 1].set_ylabel("$C_L$")
    ax[1, 1].set_xlim(left=0)

    from aerosandbox.tools.string_formatting import eng_string

    plt.suptitle(f'"{airfoil.name}" Airfoil at $Re_c = \\mathrm{{{eng_string(Re)}}}$')

    p.show_plot()
