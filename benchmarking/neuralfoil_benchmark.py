import aerosandbox as asb
import aerosandbox.numpy as np
from neuralfoil import get_aero_from_airfoil

airfoil = asb.Airfoil("dae11")

airfoil = airfoil.repanel().normalize()

alpha = np.linspace(-15, 15, 100)
Re = 1e6

aeros = {}

for model_size in ["xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge"]:
    aeros[f"NF-{model_size}"] = get_aero_from_airfoil(
        airfoil=airfoil,
        alpha=alpha,
        Re=Re,
        model_size=model_size
    )
