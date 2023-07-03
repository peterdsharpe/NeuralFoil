import torch
from typing import Union
import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
from pathlib import Path
from training.train_blind_neural_network import Net

net = Net()
net.eval()
net.load_state_dict(torch.load(Path(__file__).parent / "blind_neural_network.pth"))


def get_CL(
        airfoil: asb.Airfoil,
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
):
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    kulfan_params = get_kulfan_parameters(airfoil.coordinates, n_weights_per_side=8)

    net_inputs = np.row_stack(np.broadcast(*{
        "alpha / 10"               : alpha / 10,
        "log10_Re - 5"             : np.log10(Re) - 5,
        'kulfan_lower_0 * 5'       : kulfan_params["lower_weights"][0] * 5,
        'kulfan_lower_1 * 5'       : kulfan_params["lower_weights"][1] * 5,
        'kulfan_lower_2 * 5'       : kulfan_params["lower_weights"][2] * 5,
        'kulfan_lower_3 * 5'       : kulfan_params["lower_weights"][3] * 5,
        'kulfan_lower_4 * 5'       : kulfan_params["lower_weights"][4] * 5,
        'kulfan_lower_5 * 5'       : kulfan_params["lower_weights"][5] * 5,
        'kulfan_lower_6 * 5'       : kulfan_params["lower_weights"][6] * 5,
        'kulfan_lower_7 * 5'       : kulfan_params["lower_weights"][7] * 5,
        'kulfan_upper_0 * 5'       : kulfan_params["upper_weights"][0] * 5,
        'kulfan_upper_1 * 5'       : kulfan_params["upper_weights"][1] * 5,
        'kulfan_upper_2 * 5'       : kulfan_params["upper_weights"][2] * 5,
        'kulfan_upper_3 * 5'       : kulfan_params["upper_weights"][3] * 5,
        'kulfan_upper_4 * 5'       : kulfan_params["upper_weights"][4] * 5,
        'kulfan_upper_5 * 5'       : kulfan_params["upper_weights"][5] * 5,
        'kulfan_upper_6 * 5'       : kulfan_params["upper_weights"][6] * 5,
        'kulfan_upper_7 * 5'       : kulfan_params["upper_weights"][7] * 5,
        'kulfan_LE_weight * 5'     : kulfan_params['leading_edge_weight'] * 5,
        "kulfan_TE_thickness * 100": kulfan_params["TE_thickness"] * 100,
    }.values()))
    with torch.no_grad():
        net_outputs = net(torch.Tensor(net_inputs)).detach().numpy()

    return {
        "CL"     : net_outputs[:, 0],
        "CD"     : 10 ** (net_outputs[:, 1] - 2),
        "CM"     : net_outputs[:, 2] / 20,
        "Cpmin"  : net_outputs[:, 3] * 2,
        "Top_Xtr": net_outputs[:, 4],
        "Bot_Xtr": net_outputs[:, 5],
    }


if __name__ == '__main__':
    # airfoil = asb.Airfoil("naca4418")
    # print(get_CL(airfoil, 4, 2e6))

    from data.load_data import df_train, df_test, weights, kulfan_cols, aero_input_cols, aero_output_cols, all_cols
    import polars as pl

    df_inputs = pl.DataFrame({
        "alpha / 10"               : df_test["alpha"] / 10,
        "log10_Re - 5"             : np.log10(df_test["Re"]) - 5,
        'kulfan_lower_0 * 5'       : df_test['kulfan_lower_0'] * 5,
        'kulfan_lower_1 * 5'       : df_test['kulfan_lower_1'] * 5,
        'kulfan_lower_2 * 5'       : df_test['kulfan_lower_2'] * 5,
        'kulfan_lower_3 * 5'       : df_test['kulfan_lower_3'] * 5,
        'kulfan_lower_4 * 5'       : df_test['kulfan_lower_4'] * 5,
        'kulfan_lower_5 * 5'       : df_test['kulfan_lower_5'] * 5,
        'kulfan_lower_6 * 5'       : df_test['kulfan_lower_6'] * 5,
        'kulfan_lower_7 * 5'       : df_test['kulfan_lower_7'] * 5,
        'kulfan_upper_0 * 5'       : df_test['kulfan_upper_0'] * 5,
        'kulfan_upper_1 * 5'       : df_test['kulfan_upper_1'] * 5,
        'kulfan_upper_2 * 5'       : df_test['kulfan_upper_2'] * 5,
        'kulfan_upper_3 * 5'       : df_test['kulfan_upper_3'] * 5,
        'kulfan_upper_4 * 5'       : df_test['kulfan_upper_4'] * 5,
        'kulfan_upper_5 * 5'       : df_test['kulfan_upper_5'] * 5,
        'kulfan_upper_6 * 5'       : df_test['kulfan_upper_6'] * 5,
        'kulfan_upper_7 * 5'       : df_test['kulfan_upper_7'] * 5,
        'kulfan_LE_weight * 5'     : df_test['kulfan_LE_weight'] * 5,
        "kulfan_TE_thickness * 100": df_test["kulfan_TE_thickness"] * 100,
    })

    with torch.no_grad():
        net_outputs = net(torch.Tensor(df_inputs.to_numpy())).detach().numpy()

    df_outputs = pl.DataFrame({
        "CL_model"     : net_outputs[:, 0],
        "CD_model"     : 10 ** (net_outputs[:, 1] - 2),
        "CM_model"     : net_outputs[:, 2] / 20,
        "Cpmin_model"  : net_outputs[:, 3] * 2,
        "Top_Xtr_model": net_outputs[:, 4],
        "Bot_Xtr_model": net_outputs[:, 5],
    })
    df = pl.concat([df_test, df_outputs], how="horizontal")

    CL_mae = pl.mean((df["CL"] - df["CL_model"]).abs())
    print(f"CL MAE: {CL_mae}")