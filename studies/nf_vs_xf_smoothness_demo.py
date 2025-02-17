import aerosandbox as asb
import aerosandbox.numpy as np
from typing import Optional
from tqdm import tqdm
from dataclasses import dataclass, field
from functools import cached_property, lru_cache

init_guess_af = asb.KulfanAirfoil("naca0012")


@dataclass(frozen=True)
class State:
    """
    Defines a single aerodynamic operating condition, consisting of:
        - An airfoil
        - An angle of attack
        - A Reynolds number
        - A Mach number

    Computation of the aerodynamics is lazy, and is only computed when needed. You can compute the
    aerodynamics using the following methods:
        - `aero_nf()` for NeuralFoil
        - `aero_xf()` for XFoil

    Both will return a dictionary with the following keys (among others):
        - "CL" (lift coefficient)
        - "CD" (drag coefficient)
        - "CM" (moment coefficient)
    """
    af: asb.KulfanAirfoil
    alpha: float = 0.0
    Re: float = 1e6
    mach: float = 0.0
    _precomputed_aero_nf: Optional[dict] = field(default=None, repr=False)
    _precomputed_aero_xf: Optional[dict] = field(default=None, repr=False)

    @cached_property
    def aero_nf(self) -> dict[str, float | np.ndarray]:
        if self._precomputed_aero_nf is not None:
            return self._precomputed_aero_nf
        return self.af.get_aero_from_neuralfoil(
            alpha=self.alpha,
            Re=self.Re,
            mach=self.mach,
            model_size="xxxlarge",
        )

    @cached_property
    def aero_xf(self) -> dict[str, float | np.ndarray]:
        if self._precomputed_aero_xf is not None:
            return self._precomputed_aero_xf
        return asb.XFoil(
            self.af,
            Re=self.Re,
            mach=self.mach,
        ).alpha(self.alpha)

    def force_precompute(self) -> None:
        self.aero_nf  # Forces cached_property to compute
        self.aero_xf  # Forces cached_property to compute

    def optimize_nf(self, model_size="xlarge") -> "State":
        """
        Given the current state, formulates an aerodynamic shape optimization problem using
        NeuralFoil to minimize the drag coefficient. Uses the current state as the initial guess.

        Args:
            model_size: The size of the model to use in the NeuralFoil analysis. Options are:
                - "xxsmall"
                - "xsmall"
                - "small"
                - "medium"
                - "large"
                - "xlarge"
                - "xxlarge"
                - "xxxlarge"

        Returns: The optimized state.

        """
        opti = asb.Opti()

        af = asb.KulfanAirfoil(
            name="Optimized",
            lower_weights=opti.variable(
                init_guess=self.af.lower_weights,
                lower_bound=-1,
                upper_bound=1,
            ),
            upper_weights=opti.variable(
                init_guess=self.af.upper_weights,
                lower_bound=-1,
                upper_bound=1,
            ),
            leading_edge_weight=opti.variable(
                init_guess=self.af.leading_edge_weight,
                lower_bound=-1,
                upper_bound=1,
            ),
            TE_thickness=self.af.TE_thickness,
        )

        alpha = opti.variable(init_guess=self.alpha, lower_bound=-30, upper_bound=30)

        aero = af.get_aero_from_neuralfoil(
            alpha=alpha,
            Re=self.Re,
            mach=self.mach,
            model_size=model_size,
        )

        opti.subject_to(
            [
                aero["analysis_confidence"] > 0.95,
                aero["CL"] == 0.5,
                af.local_thickness() > 0,
                af.area() / init_guess_af.area() > 0.85,
                af.LE_radius() / init_guess_af.LE_radius() > 0.85,
                af.max_thickness() / init_guess_af.max_thickness() > 0.25,
                af.lower_weights[0] + af.upper_weights[0] == 0,
            ]
        )
        opti.minimize(aero["CD"])

        sol = opti.solve(behavior_on_failure="return_last", verbose=False)

        return State(
            af=sol(af),
            alpha=sol(alpha),
            Re=self.Re,
            mach=self.mach,
            _precomputed_aero_nf=sol(aero),
        )

    def blend_with(self, other: "State", blend_fraction: float) -> "State":
        return State(
            af=self.af.blend_with_another_airfoil(other.af, blend_fraction=blend_fraction),
            alpha=self.alpha * (1 - blend_fraction) + other.alpha * blend_fraction,
            Re=self.Re * (1 - blend_fraction) + other.Re * blend_fraction,
            mach=self.mach * (1 - blend_fraction) + other.mach * blend_fraction,
        )

s1 = State(af=asb.KulfanAirfoil("dae11"), alpha=2)
s2 = State(af=asb.KulfanAirfoil("dae11"), alpha=5)

blend_fractions = np.linspace(0, 1, 600)

s = [
    s1.blend_with(s2, blend_fraction=f)
    for f in blend_fractions
]

for state in tqdm(s):
    state.force_precompute()

import matplotlib.pyplot as plt
import aerosandbox.tools.pretty_plots as p

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(
    blend_fractions,
    np.concatenate([state.aero_nf["CD"] for state in s]),
    label="NeuralFoil",
)
ax.plot(
    blend_fractions,
    np.array([
        float(state.aero_xf["CD"]) if len(state.aero_xf["CD"]) == 1 else np.nan
        for state in s
    ]),
    label="XFoil",
)

plt.xticks(
    [0, 1],
    [
        "DAE-11\nat $\\alpha = 2\\degree$",
        "DAE-11\nat $\\alpha = 5\\degree$",
    ],
)
p.set_ticks(None, 0.1)

p.show_plot(
    "Linear Variation between Two Designs",
    "",
    "Predicted Drag Coefficient $C_D$",
    set_ticks=False,
    savefig="nf_vs_xf_smoothness_demo.svg",
)
