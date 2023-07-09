import aerosandbox as asb
import aerosandbox.numpy as np
from typing import Union


def get_CL(
        airfoil: asb.Airfoil,
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    p = {
        'CLa_low'            : 5.796551904649262,
        'CLa_high'           : 6.662038127483676,
        'log10_Re_switch'    : 4.759293563062047,
        'log10_Re_scale'     : 0.40489480755424206,
        'CL0_high_LE'        : 0.17280052131807694,
        'CL0_high_0'         : 0.31799245171528784,
        'CL0_high_1'         : 0.2826151876712608,
        'CL0_high_2'         : 0.24841615861002925,
        'CL0_high_3'         : 0.24016960597766465,
        'CL0_high_4'         : 0.23497448970552062,
        'CL0_high_5'         : 0.23179443674621486,
        'CL0_high_6'         : 0.27764987935170676,
        'CL0_high_7'         : 0.3558323531831959,
        'CL0_low_LE'         : 0.12160802332276614,
        'CL0_low_0'          : 0.11754623987319483,
        'CL0_low_1'          : 0.038035213950785456,
        'CL0_low_2'          : -0.1108768707653387,
        'CL0_low_3'          : -0.10011194827235753,
        'CL0_low_4'          : -0.019534698670369936,
        'CL0_low_5'          : 0.057508354923867314,
        'CL0_low_6'          : 0.10926620536707825,
        'CL0_low_7'          : -0.10812301223195821,
        'delta_CLa_high_area': -0.2887303436003777,
        'delta_CLa_low_area' : -1.2434390140977036
    }

    from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
    kulfan_params = get_kulfan_parameters(airfoil.coordinates, n_weights_per_side=8)

    x = {
        'alpha'              : alpha,
        'Re'                 : Re,
        **{f"kulfan_lower_{i}": kulfan_params["lower_weights"][i] for i in range(8)},
        **{f"kulfan_upper_{i}": kulfan_params["upper_weights"][i] for i in range(8)},
        "kulfan_TE_thickness": kulfan_params["TE_thickness"],
        "kulfan_LE_weight"   : kulfan_params["leading_edge_weight"]
    }

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


if __name__ == '__main__':
    af = asb.Airfoil("ag36")
    alpha = np.linspace(-10, 10, 100)
    Re = 1e6
    CL = get_CL(af, alpha, Re)
