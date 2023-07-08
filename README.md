
<p align="center">
    <img src="./media/neuralfoil.svg" width="800" />
</p>

by Peter Sharpe

-----

NeuralFoil is a small, simple tool for rapid aerodynamics analysis of airfoils. Under the hood, it consists of a neural network trained on tens of millions of XFoil runs. 



## Installation and Usage

## FAQs

Why not just use XFoil directly?

> XFoil is a truly excellent piece of aerospace software engineering and is the gold standard of airfoil analysis, for good reason. When its assumptions hold (airfoils in subsonic flow without massive separation), its accuracy exceeds that of RANS CFD, yet it has ~1000x lower computational cost. XFoil shines in particular for human-in-the-loop airfoil design. However, XFoil is not the right tool for all applications, for a few reasons:
> - XFoil exhibits hysteresis: you can get slightly different solutions (for the same airfoil, $\alpha$, and $Re$) depending on whether you sweep $\alpha$ up or down, as Newton iteration is resumed from the last converged solution and uniqueness is not guaranteed. This hysteresis can be a big problem for design optimization.
> - XFoil is not differentiable, in the sense that it doesn't come with an adjoint. That's okay (NeuralFoil doesn't either, at least out-of-the-box), but many tools exist for easily differentiating NeuralFoil's NumPy code (e.g., JAX), while options for XFoil's Fortran code either don't exist or are significantly less advanced (e.g., Tapenade). 
> - XFoil's solutions lack $C_1$-continuity. NeuralFoil, by contrast, is guaranteed to be $C_\infty$-continuous by construction. This is critical for gradient-based optimization.
>   - Even if one tries to compute gradients of XFoil's outputs by finite-differencing or complex-stepping, these gradients are often inaccurate. 
>     - A bit into the weeds, but: this comes down to how XFoil handles transition (onset of turbulence). XFoil does a cut-cell approach on the transitioning interval, and while this specific cut-cell implementation restores $C_0$-continuity (i.e., transition won't truly "jump" from one node to another discretely), gradients of closure functions still change at the cell interface - this loses $C_1$ continuity, causing a "ragged" polar at the microscopic level.
> - XFoil is not vectorized
> - XFoil is not guaranteed to produce a solution. In practice, XFoil often crashes when "ambitious" calculations are attempted, rather than producing a less-accurate answer. In some applications, that's okay or even desirable; in others, that's  



[![Build Status](https://travis-ci.org/NeuralFoil/NeuralFoil.svg?branch=master)](https://travis-ci.org/NeuralFoil/NeuralFoil)
[![codecov](https://codecov.io/gh/NeuralFoil/NeuralFoil/branch/master/graph/badge.svg)](https://codecov.io/gh/NeuralFoil/NeuralFoil)
[![Documentation Status](https://readthedocs.org/projects/neuralfoil/badge/?version=latest)](https://neuralfoil.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/neuralfoil.svg)](https://badge.fury.io/py/neuralfoil)
[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)