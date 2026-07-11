"""
Tests that the different user-facing entry points (Airfoil object, raw
coordinates, .dat file) give consistent answers for the same geometry.
"""

import numpy as np
import pytest
import aerosandbox as asb

import neuralfoil as nf

ALPHA = 4.0
RE = 2e6
SCALAR_KEYS = ["analysis_confidence", "CL", "CD", "CM", "Top_Xtr", "Bot_Xtr"]


@pytest.fixture(scope="module")
def airfoil() -> asb.Airfoil:
    return asb.Airfoil("naca4412")


def test_airfoil_vs_coordinates(airfoil):
    aero_airfoil = nf.get_aero_from_airfoil(airfoil, alpha=ALPHA, Re=RE)
    aero_coordinates = nf.get_aero_from_coordinates(
        airfoil.coordinates, alpha=ALPHA, Re=RE
    )
    for key in SCALAR_KEYS:
        np.testing.assert_allclose(
            aero_airfoil[key], aero_coordinates[key], rtol=1e-10, err_msg=key
        )


def test_airfoil_vs_dat_file(airfoil, tmp_path):
    dat_file = tmp_path / "naca4412.dat"
    dat_file.write_text(
        airfoil.name
        + "\n"
        + "\n".join(f"{x:.16f} {y:.16f}" for x, y in airfoil.coordinates)
    )

    aero_airfoil = nf.get_aero_from_airfoil(airfoil, alpha=ALPHA, Re=RE)
    aero_dat = nf.get_aero_from_dat_file(filename=dat_file, alpha=ALPHA, Re=RE)
    for key in SCALAR_KEYS:
        # Slightly looser tolerance: the .dat round-trip re-fits the geometry
        # from finite-precision text.
        np.testing.assert_allclose(
            aero_airfoil[key], aero_dat[key], rtol=1e-6, err_msg=key
        )


if __name__ == "__main__":
    pytest.main([__file__])
