import aerosandbox as asb
import aerosandbox.numpy as np


initial_guess_airfoil = asb.KulfanAirfoil("naca0012")
initial_guess_airfoil.name = "Initial Guess (NACA0012)"

opti = asb.Opti()

Re = opti.variable(init_guess=1e6, log_transform=True)
mach = 0

optimized_airfoil = asb.KulfanAirfoil(
    name="Optimized",
    lower_weights=opti.variable(
        init_guess=initial_guess_airfoil.lower_weights,
        # lower_bound=-0.5,
        # upper_bound=0.25,
    ),
    upper_weights=opti.variable(
        init_guess=initial_guess_airfoil.upper_weights,
        # lower_bound=-0.25,
        # upper_bound=0.5,
    ),
    leading_edge_weight=opti.variable(
        init_guess=initial_guess_airfoil.leading_edge_weight,
        # lower_bound=-1,
        # upper_bound=1,
    ),
    TE_thickness=0,
)

alpha = opti.variable(init_guess=5, lower_bound=0, upper_bound=180)

aero = optimized_airfoil.get_aero_from_neuralfoil(
    alpha=alpha,
    Re=Re,
    mach=mach,
    model_size="xlarge",
)

opti.subject_to(
    [
        aero["analysis_confidence"] > 0.99,
        optimized_airfoil.LE_radius() > 0.01,
        optimized_airfoil.local_thickness() > 0,
    ]
)
opti.maximize(np.sind(alpha) * aero["analysis_confidence"])

sol = opti.solve(
    behavior_on_failure="return_last",
    options={
        # "ipopt.mu_strategy": 'monotone',
        # "ipopt.start_with_resto": 'yes'
    },
)

optimized_airfoil = sol(optimized_airfoil)
aero = sol(aero)
