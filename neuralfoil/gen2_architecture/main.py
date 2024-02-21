import aerosandbox as asb
import aerosandbox.numpy as np
from typing import Union, Dict, Set, List
from pathlib import Path
from neuralfoil.gen2_architecture._basic_data_type import Data

npz_file_directory = Path(__file__).parent / "nn_weights_and_biases"


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _inverse_sigmoid(x):
    return -np.log(1 / x - 1)


def get_aero_from_kulfan_parameters(
        kulfan_parameters: Dict[str, Union[float, np.ndarray]],
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
        n_crit: Union[float, np.ndarray] = 9.0,
        xtr_upper: Union[float, np.ndarray] = 1.0,
        xtr_lower: Union[float, np.ndarray] = 1.0,
        model_size="large"
) -> Dict[str, Union[float, np.ndarray]]:

    ### Load the neural network parameters
    filename = npz_file_directory / f"nn-{model_size}.npz"
    if not filename.exists():
        raise FileNotFoundError(
            f"Could not find the neural network file {filename}, which contains the weights and biases.")

    data: Dict[str, np.ndarray] = np.load(filename)

    ### Prepare the inputs for the neural network
    input_rows: List[Union[float, np.ndarray]] = [
        *kulfan_parameters["upper_weights"],
        *kulfan_parameters["lower_weights"],
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

    x = np.stack(input_rows, axis=1)  # N_cases x N_inputs
    ##### Evaluate the neural network

    ### First, determine what the structure of the neural network is (i.e., how many layers it has) by looking at the keys.
    # These keys come from the dictionary of saved weights/biases for the specified neural network.
    try:
        layer_indices: Set[int] = set([
            int(key.split(".")[1])
            for key in data.keys()
        ])
    except TypeError:
        raise ValueError(
            f"Got an unexpected neural network file format.\n"
            f"Dictionary keys should be strings of the form 'net.0.weight', 'net.0.bias', 'net.2.weight', etc.'.\n"
            f"Instead, got keys of the form {data.keys()}.\n"
        )
    layer_indices: List[int] = sorted(list(layer_indices))

    ### Now, set up evaluation of the basic neural network.
    def net(x: np.ndarray):
        """
        Evaluates the raw network (taking in scaled inputs and returning scaled outputs).
        Input `x` dims: N_cases x N_inputs
        Output `y` dims: N_cases x N_outputs
        """
        x = np.transpose(x)
        layer_indices_to_iterate = layer_indices.copy()

        while len(layer_indices_to_iterate) != 0:
            i = layer_indices_to_iterate.pop(0)
            w = data[f"net.{i}.weight"]
            b = data[f"net.{i}.bias"]
            x = w @ x + np.reshape(b, (-1, 1))

            if len(layer_indices_to_iterate) != 0:  # Don't apply the activation function on the last layer
                x = np.tanh(x)
        x = np.transpose(x)
        return x

    y = net(x)  # N_outputs x N_cases

    ### Then, flip the inputs and evaluate the network again.
    # The goal here is to embed the invariant of "symmetry across alpha" into the network evaluation.
    # (This was also performed during training, so the network is "intended" to be evaluated this way.)

    x_flipped = x + 0.  # This is a array-api-agnostic way to force a memory copy of the array to be made.
    x_flipped[:, :8] = x[:, 8:16] * -1  # switch kulfan_lower with a flipped kulfan_upper
    x_flipped[:, 8:16] = x[:, :8] * -1  # switch kulfan_upper with a flipped kulfan_lower
    x_flipped[:, 16] *= -1  # flip kulfan_LE_weight
    x_flipped[:, 18] *= -1  # flip sin(2a)
    x_flipped[:, 23] = x[:, 24]  # flip xtr_upper with xtr_lower
    x_flipped[:, 24] = x[:, 23]  # flip xtr_lower with xtr_upper

    y_flipped = net(x_flipped)

    ### The resulting outputs will also be flipped, so we need to flip them back to their normal orientation
    y_unflipped = y_flipped + 0.  # This is a array-api-agnostic way to force a memory copy of the array to be made.
    y_unflipped[:, 1] *= -1  # CL
    y_unflipped[:, 3] *= -1  # CM
    y_unflipped[:, 4] = y_flipped[:, 5]  # switch Top_Xtr with Bot_Xtr
    y_unflipped[:, 5] = y_flipped[:, 4]  # switch Bot_Xtr with Top_Xtr
    y_unflipped[:, 6:6 + 72] = y_flipped[:, 6 + 72: 6 + 2 * 72]  # switch upper_bl_theta with lower_bl_theta
    y_unflipped[:, 6 + 72: 6 + 2 * 72] = y_flipped[:, 6:6 + 72]  # switch lower_bl_theta with upper_bl_theta

    ### Then, average the two outputs to get the "symmetric" result
    y_fused = (y + y_unflipped) / 2
    y_fused[:, 0] = _sigmoid(y_fused[:, 0])  # Analysis confidence, a binary variable

    # Unpack the neural network outputs
    sqrt_Rex_approx = [((Data.bl_x_points[i] + 1e-2) / Re) ** 0.5 for i in range(Data.N)]

    results = {
        "analysis_confidence": y_fused[:, 0],
        "CL"                 : y_fused[:, 1] / 2,
        "CD"                 : np.exp((y_fused[:, 2] - 2) * 2),
        "CM"                 : y_fused[:, 3] / 20,
        "Top_Xtr"            : y_fused[:, 4],
        "Bot_Xtr"            : y_fused[:, 5],
        **{
            f"upper_bl_theta_{i}": np.exp(y_fused[:, 6 + i]) * sqrt_Rex_approx[i]
            for i in range(Data.N)
        },
        **{
            f"upper_bl_H_{i}": np.exp(y_fused[:, 6 + Data.N + i]) * 2.6
            for i in range(Data.N)
        },
        **{
            f"upper_bl_Cp_{i}": 1 - (1 + y_fused[:, 6 + 2 * Data.N + i]) ** 2
            for i in range(Data.N)
        },
        **{
            f"lower_bl_theta_{i}": np.exp(y_fused[:, 6 + 3 * Data.N + i]) * sqrt_Rex_approx[i]
            for i in range(Data.N)
        },
        **{
            f"lower_bl_H_{i}": np.exp(y_fused[:, 6 + 4 * Data.N + i]) * 2.6
            for i in range(Data.N)
        },
        **{
            f"lower_bl_Cp_{i}": 1 - (1 + y_fused[:, 6 + 5 * Data.N + i]) ** 2
            for i in range(Data.N)
        },
    }
    return {key: np.reshape(value, -1) for key, value in results.items()}


def get_aero_from_airfoil(
        airfoil: asb.Airfoil,
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
        n_crit: Union[float, np.ndarray] = 9.0,
        xtr_upper: Union[float, np.ndarray] = 1.0,
        xtr_lower: Union[float, np.ndarray] = 1.0,
        model_size="large",
) -> Dict[str, Union[float, np.ndarray]]:

    airfoil_normalization = airfoil.normalize(return_dict=True)

    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters

    kulfan_parameters = get_kulfan_parameters(
        airfoil_normalization["airfoil"].coordinates,
        n_weights_per_side=8
    )

    return get_aero_from_kulfan_parameters(
        kulfan_parameters=kulfan_parameters,
        alpha=alpha + airfoil_normalization["rotation_angle"],
        Re=Re / airfoil_normalization["scale_factor"],
        n_crit=n_crit,
        xtr_upper=xtr_upper,
        xtr_lower=xtr_lower,
        model_size=model_size
    )


def get_aero_from_coordinates(
        coordinates: np.ndarray,
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
        n_crit: Union[float, np.ndarray] = 9.0,
        xtr_upper: Union[float, np.ndarray] = 1.0,
        xtr_lower: Union[float, np.ndarray] = 1.0,
        model_size="large",
):
    return get_aero_from_airfoil(
        airfoil=asb.Airfoil(
            coordinates=coordinates
        ),
        alpha=alpha,
        Re=Re,
        n_crit=n_crit,
        xtr_upper=xtr_upper,
        xtr_lower=xtr_lower,
        model_size=model_size
    )


def get_aero_from_dat_file(
        filename,
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
        model_size="large",
):
    with open(filename, "r") as f:
        raw_text = f.readlines()

    from aerosandbox.geometry.airfoil.airfoil_families import get_coordinates_from_raw_dat
    return get_aero_from_coordinates(
        coordinates=get_coordinates_from_raw_dat(raw_text=raw_text),
        alpha=alpha,
        Re=Re,
        model_size=model_size
    )


if __name__ == '__main__':

    # airfoil = asb.Airfoil("dae11").repanel().normalize()
    airfoil = asb.Airfoil("naca0050")
    # airfoil = asb.Airfoil("naca0012").add_control_surface(10, hinge_point_x=0.5)

    alpha = np.linspace(-18, 18, 100)
    Re = 1e6

    aeros = {}

    model_sizes = ["large"]

    for model_size in model_sizes:
        aeros[f"NF-{model_size}"] = get_aero_from_airfoil(
            airfoil=airfoil,
            alpha=alpha,
            Re=Re,
            model_size=model_size
        )

    if True:

        aeros["XFoil"] = asb.XFoil(
            airfoil=airfoil,
            Re=Re,
            max_iter=20,
            xfoil_repanel=True
        ).alpha(alpha)

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    for label, aero in aeros.items():
        if "xfoil" in label.lower():
            a = aero["alpha"]
            kwargs = dict(
                color="k"
            )
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

    plt.suptitle(f"\"{airfoil.name}\" Airfoil at $Re_c = \\mathrm{{{eng_string(Re)}}}$")

    p.show_plot()
