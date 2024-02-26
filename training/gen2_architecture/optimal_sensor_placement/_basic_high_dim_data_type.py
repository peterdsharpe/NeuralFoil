import aerosandbox as asb
import aerosandbox.numpy as np
from dataclasses import dataclass, field
from typing import Union, Sequence, List, Any
from scipy import interpolate
from neuralfoil.gen2_architecture._basic_data_type import (
    Data as LowDimData,
    compute_optimal_x_points
)


@dataclass
class Data(LowDimData):

    N = 200
    bl_x_points = compute_optimal_x_points(n_points=N)


if __name__ == '__main__':
    af = asb.Airfoil("ag41d-02r").normalize().to_kulfan_airfoil()

    datas = Data.from_xfoil(
        airfoil=af,
        alphas=[3, 5, 60],
        Re=1e6,
        mach=0,
        n_crit=9,
        xtr_upper=0.99,
        xtr_lower=0.99,
    )

    d = datas[0]

    # import aerosandbox as asb
    # import aerosandbox.numpy as np
    #
    # airfoil_database_path = asb._asb_root / "geometry" / "airfoil" / "airfoil_database"
    #
    # UIUC_airfoils = [
    #     asb.Airfoil(name=filename.stem).normalize().to_kulfan_airfoil()
    #     for filename in airfoil_database_path.glob("*.dat")
    # ]
    #
    # for af in UIUC_airfoils:
    #     print(af.name)
    #     datas = Data.from_xfoil(
    #         airfoil=af,
    #         alphas=[2, 60],
    #         Re=1e6,
    #         mach=0,
    #         n_crit=9,
    #         xtr_upper=0.99,
    #         xtr_lower=0.99,
    #     )
    #
    #     d = datas[0]
    #     # print(d.to_vector())
    #     # print(Data.from_vector(d.to_vector()))
    #     print(d == Data.from_vector(d.to_vector()))
