import aerosandbox as asb
import aerosandbox.numpy as np

gamma = 1.4


def Cp_crit(M):
    return (
        2
        / (gamma * M**2)
        * (
            ((1 + (gamma - 1) / 2 * M**2) / (1 + (gamma - 1) / 2))
            ** (gamma / (gamma - 1))
            - 1
        )
    )


# Prandtl-Glauert correction
def Cp_PG(Cp0, M):
    return Cp0 / (1 - M**2) ** 0.5


# Karman-Tsien correction
def Cp_KT(Cp0, M):
    return Cp0 / ((1 - M**2) ** 0.5 + M**2 / (1 + (1 - M**2) ** 0.5) * (Cp0 / 2))


### Laitone's rule
def Cp_L(Cp0, M):
    return Cp0 / (
        (1 - M**2) ** 0.5
        + M**2 / (1 + (1 - M**2) ** 0.5) * (Cp0 / 2) * (1 + (gamma - 1) / 2 * M**2)
    )


M = np.linspace(0.001, 0.999, 500)

### First, solve with PG
opti = asb.Opti()
Cp0 = opti.variable(init_guess=-1.5, n_vars=len(M), upper_bound=0)

opti.subject_to(
    [
        Cp_crit(M) == Cp_PG(Cp0, M),
    ]
)
sol = opti.solve()
Cp0_PG = sol(Cp0)

### Then, use the PG solution as an initial guess to solve with KT
opti = asb.Opti()
Cp0 = opti.variable(init_guess=Cp0_PG, upper_bound=0)

opti.subject_to(
    [
        Cp_crit(M) == Cp_KT(Cp0, M),
    ]
)
sol = opti.solve()
Cp0_KT = sol(Cp0)

### Then, use the PG solution as an initial guess to solve with Laitone's rule
opti = asb.Opti()
Cp0 = opti.variable(init_guess=Cp0_KT, upper_bound=0)

opti.subject_to(
    [
        Cp_crit(M) == Cp_L(Cp0, M),
    ]
)
sol = opti.solve()

### Finalize data
Cp0 = sol(Cp0)
M_crit = sol(M)

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import aerosandbox.tools.pretty_plots as p

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    # plt.plot(Cp0_PG, M, "--", label="Prandtl-Glauert")
    # plt.plot(Cp0_KT, M, "-.", label="Karman-Tsien")
    plt.plot(Cp0, M, "-", label="Laitone's Rule")

    plt.xlim(-6, 0)
    plt.ylim(0.2, 1)
    p.show_plot(
        title="Critical Mach Number vs. $C_{p0}$, using Laitone's Rule",
        xlabel="Minimum Pressure Coefficient at\nIncompressible Conditions $C_{p,\\rm min,0}$ [-]",
        ylabel="Critical\nMach\nNumber\n$M_{\\rm crit}$ [-]",
        savefig="critical_mach_vs_cp0.svg",
    )
