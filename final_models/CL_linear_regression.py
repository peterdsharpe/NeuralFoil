import aerosandbox as asb
import aerosandbox.numpy as np
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters
from typing import Union


def get_CL(
        airfoil: asb.Airfoil,
        alpha: Union[float, np.ndarray],
        Re: Union[float, np.ndarray],
):
    p = {
        'CLa_low'            : 5.604249275478053,
        'CLa_high'           : 6.415488049299007,
        'log10_Re_switch'    : 4.735823565712387,
        'log10_Re_scale'     : 0.3680365835494381,
        'CL0_high_LE'        : 0.2201983106881117,
        'CL0_high_0'         : 0.37716302695763987,
        'CL0_high_1'         : 0.30492378056972036,
        'CL0_high_2'         : 0.18108401469873767,
        'CL0_high_3'         : 0.19010901112435946,
        'CL0_high_4'         : 0.2524657232838815,
        'CL0_high_5'         : 0.28543649282884065,
        'CL0_high_6'         : 0.2996357324112084,
        'CL0_high_7'         : 0.32535000838316713,
        'CL0_low_LE'         : 0.2046173481894168,
        'CL0_low_0'          : -0.20164176943013148,
        'CL0_low_1'          : 0.3493512970499533,
        'CL0_low_2'          : -0.17326184777924256,
        'CL0_low_3'          : -0.3420467942312156,
        'CL0_low_4'          : -0.07861654388182412,
        'CL0_low_5'          : 0.19533213494840926,
        'CL0_low_6'          : 0.24504115656335435,
        'CL0_low_7'          : -0.1754011212096455,
        'delta_CLa_high_area': -0.17837496201813097,
        'delta_CLa_low_area' : -1.1624132224547112
    }

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