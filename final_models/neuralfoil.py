import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
from typing import Union, Dict
from pathlib import Path

npz_file_directory = Path(__file__).parent.parent / "training"


def get_aero_from_kulfan_parameters(
        kulfan_parameters: dict[str, Union[float, np.ndarray]],
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
        model_size="medium"
) -> Dict[str, Union[float, np.ndarray]]:
    # Load the neural network parameters
    filename = npz_file_directory / f"nn-{model_size}.npz"
    if not filename.exists():
        raise FileNotFoundError(f"Could not find the neural network file {filename}.")

    data = np.load(filename)

    # Prepare the inputs for the neural network
    alpha_scaled = alpha / 10
    Re_scaled = np.log10(Re) - 5
    lower_weights_scaled = kulfan_parameters["lower_weights"] * 5
    upper_weights_scaled = kulfan_parameters["upper_weights"] * 5
    leading_edge_weight_scaled = kulfan_parameters["leading_edge_weight"] * 5
    TE_thickness_scaled = kulfan_parameters["TE_thickness"] * 100

    input_rows = [
        alpha_scaled,
        Re_scaled,
        *[lower_weights_scaled[i] for i in range(8)],
        *[upper_weights_scaled[i] for i in range(8)],
        leading_edge_weight_scaled,
        TE_thickness_scaled
    ]

    N_inputs = 1
    for row in input_rows:
        if np.length(row) == 1:
            continue
        else:
            if N_inputs == 1:
                N_inputs = np.length(row)
            else:
                if np.length(row) != N_inputs:
                    raise ValueError(
                        f"The inputs to the neural network must all have the same length. (Conflicting lengths: {N_inputs} and {np.length(row)})"
                    )

    x = np.concatenate([ # Neural net inputs
        np.reshape(np.array(row), (1, N_inputs))
        for row in input_rows
    ], axis=0)

    # Run the neural network
    try:
        layer_indices = set([
            int(key.split(".")[1])
            for key in data.keys()
        ])
    except TypeError:
        raise ValueError(
            f"Got an unexpected neural network file format."
        )

    layer_indices = sorted(list(layer_indices))
    while len(layer_indices) != 0:
        i = layer_indices.pop(0)
        w = data[f"net.{i}.weight"]
        b = data[f"net.{i}.bias"]
        x = w @ x + np.reshape(b, (-1, 1))

        if len(layer_indices) != 0:
            x = np.tanh(x)
            
    # Unpack the neural network outputs
    return {
        "CL"     : x[0, :],
        "CD"     : 10 ** (x[1, :] - 2),
        "CM"     : x[2, :] / 20,
        "Cpmin"  : x[3, :] * 2,
        "Top_Xtr": x[4, :],
        "Bot_Xtr": x[5, :],
    }
    
    
def get_aero_from_airfoil(
        airfoil: asb.Airfoil,
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
        model_size="medium",
) -> Dict[str, Union[float, np.ndarray]]:
    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters

    kulfan_parameters = get_kulfan_parameters(
        airfoil.coordinates,
        n_weights_per_side=8
    )

    return get_aero_from_kulfan_parameters(
        kulfan_parameters=kulfan_parameters,
        alpha=alpha,
        Re=Re,
        model_size=model_size
    )


def get_aero_from_coordinates(
        coordinates: np.ndarray,
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
        model_size="medium",
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
        model_size="medium",
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
    # airfoil = asb.Airfoil("dae11").normalize()
    # alpha = 4#np.linspace(-10, 10, 100)
    # Re = 1e5
    # aero = get_aero_from_airfoil(
    #     airfoil=airfoil,
    #     alpha=alpha,
    #     Re=Re,
    #     model_size="xxlarge"
    # )
    # print(aero)

    airfoil = asb.Airfoil(
        coordinates = np.concatenate([
            asb.Airfoil("dae11").upper_coordinates(),
            asb.Airfoil("dae11").upper_coordinates()[::-1][1:] * np.array([[1, -1]]),
        ]),
    )
    alpha = 0#np.linspace(-10, 10, 100)
    Re = 1e5
    aero = get_aero_from_airfoil(
        airfoil=airfoil,
        alpha=alpha,
        Re=Re,
        model_size="xxsmall"
    )
    print(aero)