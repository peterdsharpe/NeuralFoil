import aerosandbox as asb
import aerosandbox.numpy as np
from typing import Union, Dict, Set, List
from pathlib import Path

npz_file_directory = Path(__file__).parent / "nn_weights_and_biases"


def get_aero_from_kulfan_parameters(
        kulfan_parameters: Dict[str, Union[float, np.ndarray]],
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
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
        4 * np.sind(2 * alpha),
        20 * (1 - np.cosd(alpha) ** 2),
        (np.log(Re) - 12.5) / 2,
        *[kulfan_parameters["lower_weights"][i] * 5 for i in range(8)],
        *[kulfan_parameters["upper_weights"][i] * 5 for i in range(8)],
        kulfan_parameters["leading_edge_weight"] * 5,
        kulfan_parameters["TE_thickness"] * 100,
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

    x = np.stack(input_rows, axis=0)

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
        layer_indices_to_iterate = layer_indices.copy()

        while len(layer_indices_to_iterate) != 0:
            i = layer_indices_to_iterate.pop(0)
            w = data[f"net.{i}.weight"]
            b = data[f"net.{i}.bias"]
            x = w @ x + np.reshape(b, (-1, 1))

            if len(layer_indices_to_iterate) != 0:  # Don't apply the activation function on the last layer
                x = np.tanh(x)

        return x

    y = net(x)

    ### Then, flip the inputs and evaluate the network again.
    # The goal here is to embed the invariant of "symmetry across alpha" into the network evaluation.
    # (This was also performed during training, so the network is "intended" to be evaluated this way.)
    input_rows_flipped: List[Union[float, np.ndarray]] = [
        -4 * np.sind(2 * alpha),
        20 * (1 - np.cosd(alpha) ** 2),
        (np.log(Re) - 12.5) / 2,
        *[kulfan_parameters["upper_weights"][i] * -5 for i in range(8)],
        *[kulfan_parameters["lower_weights"][i] * -5 for i in range(8)],
        kulfan_parameters["leading_edge_weight"] * -5,
        kulfan_parameters["TE_thickness"] * 100,
    ]

    for i, row in enumerate(input_rows_flipped):
        input_rows_flipped[i] = np.ones(N_cases) * row

    x_flipped = np.stack(input_rows_flipped, axis=0)

    y_flipped = net(x_flipped)

    ### The resulting outputs will also be flipped, so we need to flip them back to their normal orientation
    y_flipped[0, :] *= -1  # CL
    y_flipped[2, :] *= -1  # CM
    temp = y_flipped[4, :] + 0.  # This is here to help swap the top / bottom Xtr (transition x) values
    # Adding the 0. is a hack to force a memory-copy of the array to be made in a way that's array-backend-agnostic.
    y_flipped[4, :] = y_flipped[5,:]  # Replace top Xtr with bottom Xtr
    y_flipped[5, :] = temp  # Replace bottom Xtr with top Xtr

    ### Then, average the two outputs to get the "symmetric" result
    y = (y + y_flipped) / 2

    # Unpack the neural network outputs
    results = {
        "CL"     : y[0, :],
        "CD"     : np.exp(y[1, :] - 4),
        "CM"     : y[2, :] / 20,
        "Cpmin"  : 1 - y[3, :] ** 2,
        "Top_Xtr": y[4, :],
        "Bot_Xtr": y[5, :],
    }
    return {key: np.reshape(value, -1) for key, value in results.items()}


def get_aero_from_airfoil(
        airfoil: asb.Airfoil,
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
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
        model_size=model_size
    )


def get_aero_from_coordinates(
        coordinates: np.ndarray,
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
        model_size="large",
):
    return get_aero_from_airfoil(
        airfoil=asb.Airfoil(
            coordinates=coordinates
        ),
        alpha=alpha,
        Re=Re,
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

    airfoil = asb.Airfoil("dae11").repanel().normalize()
    # airfoil = asb.Airfoil("naca0012").add_control_surface(10, hinge_point_x=0.5)

    alpha = np.linspace(-15, 15, 100)
    Re = 1e6

    aeros = {}

    model_sizes = ["xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge"]

    for model_size in model_sizes:
        aeros[f"NF-{model_size}"] = get_aero_from_airfoil(
            airfoil=airfoil,
            alpha=alpha,
            Re=Re,
            model_size=model_size
        )

    if False:

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
