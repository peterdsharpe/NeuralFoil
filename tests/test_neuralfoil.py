import pytest
import neuralfoil as nf
import aerosandbox as asb


def test_basic_functionality():
    nf.get_aero_from_airfoil(asb.Airfoil("naca4412"), alpha=5, Re=1e6)


if __name__ == "__main__":
    pytest.main()
