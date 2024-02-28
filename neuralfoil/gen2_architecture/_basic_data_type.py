import warnings
import aerosandbox as asb
import aerosandbox.numpy as np
from dataclasses import dataclass, field
from typing import Union, Sequence, List, Any
from scipy import interpolate


def compute_optimal_x_points(
        n_points,
):
    # theta = np.arccos(1 - 2 * x)
    # delta_Cp_thin_airfoil_theory = 4 / np.tan(theta)
    # desired_spacing = 1 / delta_Cp_thin_airfoil_theory
    # spacing_function = integral(desired_spacing, x) # rescaled to (0, 1)

    # thin_airfoil_theory_spacing_function = lambda x: (
    #                                                          -np.sqrt((1 - x) * x)
    #                                                          - 0.5 * np.arcsin(1 - 2 * x)
    #                                                  ) / (np.pi / 2) + 0.5
    #
    # uniform_spacing_function = lambda x: x
    #
    # return (
    #         thin_airfoil_theory_spacing_function(np.linspace(0, 1, n_points)) +
    #         uniform_spacing_function(np.sinspace(0, 1, n_points))
    # ) / 2
    s = np.linspace(0, 1, n_points + 1)
    return (s[1:] + s[:-1]) / 2


# def compute_optimal_x_points(
#         n_points: int = 24,
# ):
#     # theta = np.arccos(1 - 2 * x)
#     # delta_Cp_thin_airfoil_theory = 4 / np.tan(theta)
#     # desired_spacing = 1 / delta_Cp_thin_airfoil_theory
#     # spacing_function = integral(desired_spacing, x) # rescaled to (0, 1)
#
#     thin_airfoil_theory_spacing_function = lambda x: (
#                                                              -np.sqrt((1 - x) * x)
#                                                              - 0.5 * np.arcsin(1 - 2 * x)
#                                                      ) / (np.pi / 2) + 0.5
#
#     uniform_spacing_function = lambda x: x
#
#     return (
#             thin_airfoil_theory_spacing_function(np.linspace(0, 1, n_points)) +
#             uniform_spacing_function(np.sinspace(0, 1, n_points))
#     ) / 2


@dataclass
class Data():
    airfoil: asb.KulfanAirfoil
    alpha: float
    Re: float
    mach: float
    n_crit: float
    xtr_upper: float
    xtr_lower: float

    N = 32
    bl_x_points = compute_optimal_x_points(n_points=N)

    analysis_confidence: float  # Nominally 0 (no confidence) to 1 (high confidence)
    af_outputs: dict[str, Any] = field(
        default_factory=lambda: {
            "CL"     : np.nan,
            "CD"     : np.nan,
            "CM"     : np.nan,
            "Top_Xtr": np.nan,
            "Bot_Xtr": np.nan,
        }
    )
    upper_bl_outputs: dict[str, Any] = field(
        default_factory=lambda: {
            "theta"  : np.nan * np.ones_like(Data.bl_x_points),
            "H"      : np.nan * np.ones_like(Data.bl_x_points),
            "ue/vinf": np.nan * np.ones_like(Data.bl_x_points),
        }
    )  # theta, H, ue/vinf
    lower_bl_outputs: dict[str, Any] = field(
        default_factory=lambda: {
            "theta"  : np.nan * np.ones_like(Data.bl_x_points),
            "H"      : np.nan * np.ones_like(Data.bl_x_points),
            "ue/vinf": np.nan * np.ones_like(Data.bl_x_points),
        }
    )  # theta, H, ue/vinf

    @property
    def inputs(self):
        return {
            "airfoil"  : self.airfoil,
            "alpha"    : self.alpha,
            "Re"       : self.Re,
            "mach"     : self.mach,
            "n_crit"   : self.n_crit,
            "xtr_upper": self.xtr_upper,
            "xtr_lower": self.xtr_lower,
        }

    @property
    def outputs(self):
        return {
            "analysis_confidence": self.analysis_confidence,
            "af_outputs"         : self.af_outputs,
            "upper_bl_outputs"   : self.upper_bl_outputs,
            "lower_bl_outputs"   : self.lower_bl_outputs,
        }

    @classmethod
    def from_xfoil(cls,
                   airfoil: Union[asb.Airfoil, asb.KulfanAirfoil],
                   alphas: Union[float, Sequence[float]],
                   Re: float,
                   mach: float,
                   n_crit: float,
                   xtr_upper: float,
                   xtr_lower: float,
                   timeout=5,
                   max_iter=100,
                   xfoil_command: str = 'xfoil'
                   ) -> List["Data"]:
        airfoil = airfoil.normalize().to_kulfan_airfoil()

        alphas = np.atleast_1d(alphas)

        xf = asb.XFoil(
            airfoil=airfoil,
            Re=Re,
            mach=mach,
            n_crit=n_crit,
            xtr_upper=xtr_upper,
            xtr_lower=xtr_lower,
            xfoil_repanel=True,
            timeout=timeout,
            max_iter=max_iter,
            xfoil_command=xfoil_command,
            include_bl_data=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            xf_outputs = xf.alpha(alphas)

        training_datas = []

        def append_empty_data():
            training_datas.append(
                cls(
                    airfoil=airfoil.to_kulfan_airfoil(),
                    alpha=alpha,
                    Re=Re,
                    mach=mach,
                    n_crit=n_crit,
                    xtr_upper=xtr_upper,
                    xtr_lower=xtr_lower,
                    analysis_confidence=0,
                )
            )

        for i, alpha in enumerate(alphas):
            ### Figure out which output corresponds to this alpha
            alpha_deviations = np.abs(xf_outputs["alpha"] - alpha)

            if (  # If the alpha is not in the output
                    (len(alpha_deviations) == 0) or
                    (np.min(alpha_deviations) > 0.001)
            ):
                append_empty_data()
                continue

            index = np.argmin(alpha_deviations)
            xf_output = {
                key: value[index] for key, value in xf_outputs.items()
            }
            bl_data = xf_output["bl_data"]

            # ### Trim off the wake data (rows where "Cp" first becomes NaN)
            # wake_node_indices = np.flatnonzero(bl_data["x"] > )
            # bl_data = bl_data.iloc[
            #           :np.flatnonzero(bl_data["x"] <= airfoil.x().max())[-1] + 1,
            #           :
            #           ]

            ### Split the boundary layer data into upper and lower sections
            # LE_index = bl_data["x"].argmin()
            try:
                dx = np.diff(bl_data["x"].values)
                upper_bl_data = bl_data.iloc[
                                :np.flatnonzero(dx < 0)[-1] + 2
                                :
                                ].iloc[::-1, :]
                lower_bl_data = bl_data.iloc[
                                np.flatnonzero(dx > 0)[0]:,
                                :
                                ]
            except IndexError as e:  # If the boundary layer data is too short
                print(e)
                append_empty_data()
                continue

            if len(upper_bl_data) <= 4 or len(lower_bl_data) <= 4:  # If the boundary layer data is too short
                print("BL data too short")
                append_empty_data()
                continue

            interp = lambda x, y: interpolate.PchipInterpolator(x, y, extrapolate=True)(cls.bl_x_points)

            try:
                training_datas.append(
                    cls(
                        airfoil=airfoil.to_kulfan_airfoil(),
                        alpha=alpha,
                        Re=Re,
                        mach=mach,
                        n_crit=n_crit,
                        xtr_upper=xtr_upper,
                        xtr_lower=xtr_lower,
                        analysis_confidence=1,
                        af_outputs={
                            "CL"     : xf_output["CL"],
                            "CD"     : xf_output["CD"],
                            "CM"     : xf_output["CM"],
                            "Top_Xtr": xf_output["Top_Xtr"],
                            "Bot_Xtr": xf_output["Bot_Xtr"],
                        },
                        upper_bl_outputs={
                            "theta"  : interp(upper_bl_data["x"], upper_bl_data["theta"]),
                            "H"      : interp(upper_bl_data["x"], upper_bl_data["H"]),
                            "ue/vinf": interp(upper_bl_data["x"], upper_bl_data["ue/vinf"]),
                        },
                        lower_bl_outputs={
                            "theta"  : interp(lower_bl_data["x"], lower_bl_data["theta"]),
                            "H"      : interp(lower_bl_data["x"], lower_bl_data["H"]),
                            "ue/vinf": interp(lower_bl_data["x"], lower_bl_data["ue/vinf"]),
                        },
                    )
                )
            except ValueError as e:
                print(e)
                append_empty_data()
                continue

        return training_datas

    def to_vector(self) -> np.ndarray:  # dtype: float32
        items = [
            self.airfoil.upper_weights,
            self.airfoil.lower_weights,
            self.airfoil.leading_edge_weight,
            self.airfoil.TE_thickness,
            self.alpha,
            self.Re,
            self.mach,
            self.n_crit,
            self.xtr_upper,
            self.xtr_lower,
            self.analysis_confidence,
            self.af_outputs["CL"],
            self.af_outputs["CD"],
            self.af_outputs["CM"],
            self.af_outputs["Top_Xtr"],
            self.af_outputs["Bot_Xtr"],
            self.upper_bl_outputs["theta"],
            self.upper_bl_outputs["H"],
            self.upper_bl_outputs["ue/vinf"],
            self.lower_bl_outputs["theta"],
            self.lower_bl_outputs["H"],
            self.lower_bl_outputs["ue/vinf"],
        ]

        return np.concatenate([
            np.atleast_1d(item).flatten().astype(np.float32)
            for item in items
        ], axis=0)

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "Data":
        i = 0

        def pop(n):
            nonlocal i
            result = vector[i:i + n]
            i += n
            return result

        airfoil = asb.KulfanAirfoil(
            upper_weights=pop(8),
            lower_weights=pop(8),
            leading_edge_weight=pop(1),
            TE_thickness=pop(1),
        )
        alpha = pop(1)
        Re = pop(1)
        mach = pop(1)
        n_crit = pop(1)
        xtr_upper = pop(1)
        xtr_lower = pop(1)
        analysis_confidence = pop(1)
        af_outputs = {
            "CL"     : pop(1),
            "CD"     : pop(1),
            "CM"     : pop(1),
            "Top_Xtr": pop(1),
            "Bot_Xtr": pop(1),
        }
        upper_bl_outputs = {
            "theta"  : pop(cls.N),
            "H"      : pop(cls.N),
            "ue/vinf": pop(cls.N),
        }
        lower_bl_outputs = {
            "theta"  : pop(cls.N),
            "H"      : pop(cls.N),
            "ue/vinf": pop(cls.N),
        }
        if not i == len(vector):
            raise ValueError(f"Vector length mismatch: Got {len(vector)}, expected {i}.")

        return cls(
            airfoil=airfoil,
            alpha=alpha,
            Re=Re,
            mach=mach,
            n_crit=n_crit,
            xtr_upper=xtr_upper,
            xtr_lower=xtr_lower,
            analysis_confidence=analysis_confidence,
            af_outputs=af_outputs,
            upper_bl_outputs=upper_bl_outputs,
            lower_bl_outputs=lower_bl_outputs,
        )

    @classmethod
    def get_vector_input_column_names(cls):
        return [
            *[f"kulfan_upper_{i}" for i in range(8)],
            *[f"kulfan_lower_{i}" for i in range(8)],
            "kulfan_LE_weight",
            "kulfan_TE_thickness",
            "alpha",
            "Re",
            "mach",
            "n_crit",
            "xtr_upper",
            "xtr_lower",
        ]

    @classmethod
    def get_vector_output_column_names(cls):
        return [
            "analysis_confidence",
            "CL",
            "CD",
            "CM",
            "Top_Xtr",
            "Bot_Xtr",
            *[f"upper_bl_theta_{i}" for i in range(cls.N)],
            *[f"upper_bl_H_{i}" for i in range(cls.N)],
            *[f"upper_bl_ue/vinf_{i}" for i in range(cls.N)],
            *[f"lower_bl_theta_{i}" for i in range(cls.N)],
            *[f"lower_bl_H_{i}" for i in range(cls.N)],
            *[f"lower_bl_ue/vinf_{i}" for i in range(cls.N)],
        ]

    @classmethod
    def get_vector_column_names(cls):
        return cls.get_vector_input_column_names() + cls.get_vector_output_column_names()

    def __eq__(self, other: "Data") -> bool:
        v1 = self.to_vector()
        v2 = other.to_vector()

        return all([
            np.allclose(
                s1, s2,
                atol=0,
                rtol=np.finfo(np.float32).eps * 10,
            ) or np.all(np.isnan([s1, s2]))
            for s1, s2 in zip(v1, v2)
        ])

        return np.allclose(
            self.to_vector(),
            other.to_vector(),
            atol=0,
            rtol=np.finfo(np.float32).eps * 10,
        )

    def validate_vector_format(self):
        assert self == Data.from_vector(self.to_vector())


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
