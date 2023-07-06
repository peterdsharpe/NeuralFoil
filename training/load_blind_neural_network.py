import torch
from typing import Union
import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
from pathlib import Path
from training.train_blind_neural_network import Net

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

net = Net()
net.eval()
net.load_state_dict(torch.load(Path(__file__).parent / "nn-xxlarge.pth"))


def get_aero(
        airfoil: asb.Airfoil,
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
):

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

    airfoil = asb.Airfoil("dae11").normalize()
    alpha = 4#np.linspace(-10, 10, 100)
    Re = 1e6
    aero = get_aero(
        airfoil=airfoil,
        alpha=alpha,
        Re=Re,
    )
    print(aero)
    # from data.load_data import df_train, df_test, weights, kulfan_cols, aero_input_cols, aero_output_cols, all_cols
    # import polars as pl
    #
    # for name, df_eval in {
    #     "df_test" : df_test,
    #     "df_train": df_train,
    # }.items():
    #
    #     df_inputs = pl.DataFrame({
    #         "alpha / 10"               : df_eval["alpha"] / 10,
    #         "log10_Re - 5"             : np.log10(df_eval["Re"]) - 5,
    #         'kulfan_lower_0 * 5'       : df_eval['kulfan_lower_0'] * 5,
    #         'kulfan_lower_1 * 5'       : df_eval['kulfan_lower_1'] * 5,
    #         'kulfan_lower_2 * 5'       : df_eval['kulfan_lower_2'] * 5,
    #         'kulfan_lower_3 * 5'       : df_eval['kulfan_lower_3'] * 5,
    #         'kulfan_lower_4 * 5'       : df_eval['kulfan_lower_4'] * 5,
    #         'kulfan_lower_5 * 5'       : df_eval['kulfan_lower_5'] * 5,
    #         'kulfan_lower_6 * 5'       : df_eval['kulfan_lower_6'] * 5,
    #         'kulfan_lower_7 * 5'       : df_eval['kulfan_lower_7'] * 5,
    #         'kulfan_upper_0 * 5'       : df_eval['kulfan_upper_0'] * 5,
    #         'kulfan_upper_1 * 5'       : df_eval['kulfan_upper_1'] * 5,
    #         'kulfan_upper_2 * 5'       : df_eval['kulfan_upper_2'] * 5,
    #         'kulfan_upper_3 * 5'       : df_eval['kulfan_upper_3'] * 5,
    #         'kulfan_upper_4 * 5'       : df_eval['kulfan_upper_4'] * 5,
    #         'kulfan_upper_5 * 5'       : df_eval['kulfan_upper_5'] * 5,
    #         'kulfan_upper_6 * 5'       : df_eval['kulfan_upper_6'] * 5,
    #         'kulfan_upper_7 * 5'       : df_eval['kulfan_upper_7'] * 5,
    #         'kulfan_LE_weight * 5'     : df_eval['kulfan_LE_weight'] * 5,
    #         "kulfan_TE_thickness * 100": df_eval["kulfan_TE_thickness"] * 100,
    #     })
    #
    #     with torch.no_grad():
    #         net_outputs = net(torch.Tensor(df_inputs.to_numpy())).detach().numpy()
    #
    #     df_outputs = pl.DataFrame({
    #         "CL_model"     : net_outputs[:, 0],
    #         "CD_model"     : 10 ** (net_outputs[:, 1] - 2),
    #         "CM_model"     : net_outputs[:, 2] / 20,
    #         "Cpmin_model"  : net_outputs[:, 3] * 2,
    #         "Top_Xtr_model": net_outputs[:, 4],
    #         "Bot_Xtr_model": net_outputs[:, 5],
    #     })
    #     df = pl.concat([df_eval, df_outputs], how="horizontal")
    #     print(f"\nPerformance on {name}:\n" + "-" * 20)
    #     print(f"CL MAE: {pl.median((df['CL'] - df['CL_model']).abs())}")
    #     print(f"CD MAE: {pl.median((df['CD'] - df['CD_model']).abs())}")
    #     print(f"CM MAE: {pl.median((df['CM'] - df['CM_model']).abs())}")
    #     print(f"Cpmin MAE: {pl.median((df['Cpmin'] - df['Cpmin_model']).abs())}")
    #     print(f"Top_Xtr MAE: {pl.median((df['Top_Xtr'] - df['Top_Xtr_model']).abs())}")
    #     print(f"Bot_Xtr MAE: {pl.median((df['Bot_Xtr'] - df['Bot_Xtr_model']).abs())}")
    #
    #     ##### Plotting #####
    #
    #     import matplotlib.pyplot as plt
    #     import aerosandbox.tools.pretty_plots as p
    #
    #     fig, ax = plt.subplots()
    #
    #     af = asb.Airfoil("dae11").normalize()
    #     from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
    #
    #     kulfan_params = get_kulfan_parameters(af.coordinates, n_weights_per_side=8)
    #
    #     alpha_plot = np.linspace(-10, 10, 21)
    #     Res = 1e6 * 2. ** np.arange(-5, 5)
    #     colors = plt.get_cmap("rainbow")(np.linspace(0, 1, len(Res)))
    #     colors = [p.adjust_lightness(c, 0.6) for c in colors]
    #
    #     from tqdm import tqdm
    #     from aerosandbox.tools.string_formatting import eng_string
    #
    #     for Re, color in tqdm(zip(Res, colors), desc="Sweeping Reynolds Numbers...", total=len(Res), unit="runs"):
    #         try:
    #             xf_aero = asb.XFoil(
    #                 airfoil=af,
    #                 Re=Re,
    #                 timeout=10,
    #             ).alpha(alpha_plot, start_at=4)
    #
    #             plt.plot(
    #                 xf_aero["alpha"],
    #                 xf_aero["CL"],
    #                 "-", color=color, label=eng_string(Re), alpha=0.6
    #             )
    #         except FileNotFoundError:
    #             pass
    #
    #         fit_aero = get_aero(
    #             af,
    #             alpha=alpha_plot,
    #             Re=Re,
    #         )
    #         plt.plot(
    #             alpha_plot,
    #             fit_aero["CL"],
    #             ":", color=color, alpha=0.6
    #         )
    #     plt.plot([], [], "-k", label="XFoil")
    #     plt.plot([], [], ":k", label="Linear Model")
    #     plt.xlim(alpha_plot.min(), alpha_plot.max())
    #     plt.ylim(-1.5, 1.5)
    #     p.set_ticks(2, 1, 0.5, 0.1)
    #     plt.xlabel("Angle of Attack $\\alpha$ [deg]")
    #     plt.ylabel("Lift Coefficient $C_L$ [-]")
    #     plt.title(f"{af.name} Airfoil")
    #     plt.legend(
    #         title="Reynolds Numbers", fontsize=10, ncols=2, loc="lower right"
    #     )
    #
    #     axaf = ax.inset_axes(bounds=(0.05, 0.7, 0.25, 0.25))
    #     axaf.plot(
    #         af.x(), af.y(), "-k", alpha=0.8, zorder=100
    #     )
    #     axaf.set_aspect("equal")
    #     axaf.xaxis.set_ticklabels([])
    #     axaf.yaxis.set_ticklabels([])
    #
    #     p.show_plot(legend=False)
