import aerosandbox as asb
import aerosandbox.numpy as np


initial_guess_airfoil = asb.KulfanAirfoil("naca0012")
initial_guess_airfoil.name = "Initial Guess (NACA0012)"

opti = asb.Opti()

Re = 1e6
mach=0

af = asb.KulfanAirfoil(
    name="Optimized",
    lower_weights=opti.variable(
        init_guess=initial_guess_airfoil.lower_weights,
        lower_bound=-1,
        upper_bound=0.25,
    ),
    upper_weights=opti.variable(
        init_guess=initial_guess_airfoil.upper_weights,
        lower_bound=-0.25,
        upper_bound=1,
    ),
    leading_edge_weight=opti.variable(
        init_guess=initial_guess_airfoil.leading_edge_weight,
        lower_bound=-1,
        upper_bound=1,
    ),
    TE_thickness=0,
)

alpha = opti.variable(
    init_guess=5,
    lower_bound=-30,
    upper_bound=30
)

aero = af.get_aero_from_neuralfoil(
    alpha=alpha,
    Re=Re,
    mach=mach,
    model_size="xlarge",
)

opti.subject_to([
    aero["analysis_confidence"] > 0.80,
    aero["CL"] == 0,
    # af.LE_radius() > 0.01,
    af.local_thickness() > 0
])
opti.maximize(aero["CM"] * aero["analysis_confidence"] ** 2)

sol = opti.solve(
    behavior_on_failure="return_last",
    options={
        # "ipopt.mu_strategy": 'monotone',
        # "ipopt.start_with_resto": 'yes'
    }
)

af = sol(af)
aero = sol(aero)
print(f"Re: {sol(Re)}")
print(f"alpha: {sol(alpha)}")
