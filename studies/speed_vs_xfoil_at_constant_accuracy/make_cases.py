import aerosandbox as asb
import aerosandbox.numpy as np

Thickness, Camber, Re = np.meshgrid(
    np.arange(0.08, 0.161, 0.01),
    np.arange(0.0, 0.061, 0.01),
    np.array([500e3, 2e6, 8e6]),
)

thickness_f, camber_f, Re_f = Thickness.flatten(), Camber.flatten(), Re.flatten()

afs = [
    asb.KulfanAirfoil(
        name=f"NACA{np.round(100 * c).astype(int):1d}4{np.round(100 * t).astype(int):02d}",
    )
    for c, t in zip(camber_f, thickness_f)
]

alpha = 5
