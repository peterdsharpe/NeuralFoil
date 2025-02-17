import aerosandbox as asb
from typing import NamedTuple

Re = 1e6
mach = 0


class State(NamedTuple):
    af: asb.KulfanAirfoil
    alpha: float
    aero: dict = None

state = State(
    af=asb.KulfanAirfoil("naca0012"),
    alpha=5
)


def optimize(
        state: State,
        model_size: str,
) -> State:
    opti = asb.Opti()

    af = asb.KulfanAirfoil(
        name="Optimized",
        lower_weights=opti.variable(
            init_guess=state.af.lower_weights,
            lower_bound=-1,
            upper_bound=1,
        ),
        upper_weights=opti.variable(
            init_guess=state.af.upper_weights,
            lower_bound=-1,
            upper_bound=1,
        ),
        leading_edge_weight=opti.variable(
            init_guess=state.af.leading_edge_weight,
            lower_bound=-1,
            upper_bound=1,
        ),
        TE_thickness=state.af.TE_thickness,
    )

    alpha = opti.variable(
        init_guess=state.alpha,
        lower_bound=-30,
        upper_bound=30
    )

    aero = af.get_aero_from_neuralfoil(
        alpha=alpha,
        Re=Re,
        mach=mach,
        model_size=model_size,
    )

    opti.subject_to([
        aero["analysis_confidence"] > 0.95,
        af.local_thickness() > 0
    ])
    opti.maximize(aero["CL"] / aero["CD"])

    sol = opti.solve(behavior_on_failure="return_last", verbose=False)

    print(f"Model size: {model_size}")
    xf = asb.XFoil(sol(af), Re=Re)
    xf_aero = xf.alpha(sol(alpha))
    print(f"CL/CD (XFoil): {xf_aero['CL'] / xf_aero['CD']}")
    sol(af).draw()

    return State(
        af=sol(af),
        alpha=sol(alpha),
        aero=sol(aero)
    )

state = optimize(state, model_size="small")
state = optimize(state, model_size="medium")
state = optimize(state, model_size="large")
state = optimize(state, model_size="xlarge")
state = optimize(state, model_size="xxlarge")
state = optimize(state, model_size="xxxlarge")

af = state.af
alpha = state.alpha
aero= state.aero

print(f"Re: {Re}")
print(f"alpha: {alpha}")
print(f"CL/CD: {aero['CL'] / aero['CD']}")

xf = asb.XFoil(af, Re=Re)
xf_aero = xf.alpha(alpha)
print(f"CL/CD (XFoil): {xf_aero['CL'] / xf_aero['CD']}")

af.draw()