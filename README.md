<p align="center">
    <img src="./media/neuralfoil_logo.svg" width="800" />
</p>

by [Peter Sharpe](https://peterdsharpe.github.io) (<pds [at] mit [dot] edu>)

[![Downloads](https://pepy.tech/badge/neuralfoil)](https://pepy.tech/project/neuralfoil)
[![Monthly Downloads](https://pepy.tech/badge/neuralfoil/month)](https://pepy.tech/project/neuralfoil)
[![Build Status](https://github.com/peterdsharpe/NeuralFoil/workflows/Tests/badge.svg)](https://github.com/peterdsharpe/NeuralFoil/actions/workflows/run-pytest.yml)
[![PyPI](https://img.shields.io/pypi/v/neuralfoil)](https://pypi.org/project/NeuralFoil/)
[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT)

-----

**NeuralFoil** is a tool for rapid aerodynamics analysis of airfoils, similar to [XFoil](https://web.mit.edu/drela/Public/web/xfoil/). NeuralFoil is [a hybrid of physics-informed machine learning techniques and analytical models, leveraging domain knowledge](./paper/out/main.pdf). Its learned core is trained on [tens of millions of XFoil runs](#geometry-parameterization-and-training-data).

NeuralFoil is available here as a pure Python+NumPy standalone (trained in PyTorch, runtime-executed in NumPy), but it is also [available within AeroSandbox](#extended-features-transonics-post-stall-control-surface-deflections), which extends it with advanced features. With this extension, NeuralFoil can give you **viscous, compressible airfoil aerodynamics for (nearly) any airfoil, with control surface deflections, across $360^\circ$ angle of attack, at any Reynolds number, all very quickly** (~5 milliseconds). And, it's guaranteed to return an answer (no non-convergence issues), it's vectorized, and it's $C^\infty$-continuous (critical for gradient-based optimization). For aerodynamics experts: NeuralFoil will also give you fine-grained boundary layer control ($N_{\rm crit}$, forced trips) and information ($\theta$, $H$, $u_e/V_\infty$, and pressure distributions).

A unique feature is that NeuralFoil also assesses its own trustworthiness, yielding an [`"analysis_confidence"`](#accuracy) output: queries where flow is sensitive or strongly out-of-distribution are flagged. This is especially useful for design optimization, where [constraining this uncertainty metric](https://github.com/peterdsharpe/AeroSandbox/blob/master/tutorial/06%20-%20Aerodynamics/02%20-%20AeroSandbox%202D%20Aerodynamics%20Tools/02%20-%20NeuralFoil%20Optimization.ipynb) ensures designs are [robust to small changes in shape and flow conditions.](https://web.mit.edu/drela/OldFiles/Public/papers/Pros_Cons_Airfoil_Optimization.pdf)

NeuralFoil is [~30x faster than XFoil for a single analysis, and ~1000x faster for multipoint analysis](#speed), all with [minimal loss in accuracy compared to XFoil](#accuracy). Due to the diversity of training data and the embedding of several physics-based invariants, [this accuracy is seen even on out-of-distribution airfoils](#accuracy) (i.e., airfoils it wasn't trained on). More comparisons to XFoil are [here](#xfoil-benefit-question). NeuralFoil aims to be lightweight, with [minimal dependencies](#dependencies-question) and a [small and easily-understood code-base](neuralfoil/main.py) (<500 lines of user-facing code).

```
pip install neuralfoil
```

![input-output diagram](./media/io_diagram/neuralfoil_io.png)

(The above figure is an excerpt from the [author's PhD thesis](#citing-neuralfoil))

**[For example usage of NeuralFoil, see here](https://github.com/peterdsharpe/AeroSandbox/tree/master/tutorial/06%20-%20Aerodynamics/02%20-%20AeroSandbox%202D%20Aerodynamics%20Tools).**

**[For more technical details, validation cases, and case studies, see the pre-print of the NeuralFoil paper](./paper/out/main.pdf).** ([Citation info here](#citing-neuralfoil)).

## Overview

NeuralFoil comes with 8 different neural network models, with increasing levels of complexity:

<div align="center">
    <table>
     <tr>
        <td>"xxsmall"</td>
        <td>"xsmall"</td>
        <td>"small"</td>
        <td>"medium"</td>
        <td>"large"</td>
        <td>"xlarge"</td>
        <td>"xxlarge"</td>
        <td>"xxxlarge"</td>
     </tr>
    </table>
</div>

This spectrum offers a tradeoff between accuracy and computational cost.

In addition to its neural network models, NeuralFoil also has a bonus "Linear $C_L$ model" that predicts lift coefficient $C_L$ as a purely-affine function of angle of attack $\alpha$ (though it is not affine with respect to the shape variables). This model is well-suited for linear lifting-line or blade-element-method analyses, where the $C_L(\alpha)$ linearity can be used to solve the resulting system of equations "in one shot" as a linear solve, rather than a less-numerically-robust iterative nonlinear solve.

Using NeuralFoil is dead-simple, and also offers several possible "entry points" for inputs. Here's an example showing this:

```python
import neuralfoil as nf  # `pip install neuralfoil`
import numpy as np

aero = nf.get_aero_from_dat_file(  # You can use a .dat file as an entry point
    dat_file_path="/path/to/my_airfoil_file.dat",
    alpha=5,  # Angle of attack [deg]
    Re=5e6,  # Reynolds number [-]
    model_size="xlarge",  # Optionally, specify your model size.
)

aero = nf.get_aero_from_coordinates(  # You can use xy airfoil coordinates as an entry point
    coordinates=n_by_2_numpy_ndarray_of_airfoil_coordinates,
    alpha=np.linspace(-25, 25, 1000),  # Vectorize your evaluations across `alpha` and `Re`
    Re=5e6,
)

import aerosandbox as asb  # `pip install aerosandbox`
aero = nf.get_aero_from_airfoil(  # You can use AeroSandbox airfoils as an entry point
    airfoil=asb.Airfoil("naca4412"),  # any UIUC or NACA airfoil name works
    alpha=5, Re=5e6,
)

# `aero` is a dictionary with keys: ["analysis_confidence", "CL", "CD", "CM", "Top_Xtr", "Bot_Xtr", ...]
```

## Performance

### Accuracy

Qualitatively, NeuralFoil tracks XFoil very closely across a wide range of $\alpha$ and $Re$ values. In the figure below, we compare the performance of NeuralFoil to XFoil on $C_L, C_D$ polar prediction. Notably, the airfoil analyzed here was developed "from scratch" for a [real-world aircraft development program](https://www.prnewswire.com/news-releases/electra-flies-solar-electric-hybrid-research-aircraft-301633713.html) and is completely separate from [the airfoils used during NeuralFoil's training](#geometry-parameterization-and-training-data), so NeuralFoil isn't cheating by "memorizing" this airfoil's performance. Each color in the figure below represents analyses at a different Reynolds number.

<a name="clcd-polar"></a>

<p align="center">
    <img src="./benchmarking/neuralfoil_point_comparison.svg" width="1000" />
</p>

NeuralFoil is typically accurate to within a few percent of XFoil's predictions. Note that this figure is on a truly out-of-sample airfoil, so airfoils that are closer to the training set will have even more accurate results.

NeuralFoil also [has the benefit of smoothing out XFoil's "jagged" predictions](#xfoil-benefit-question) (for example, near $C_L=1.4$ at $Re=\mathrm{80k}$) in cases where XFoil is not reliably converging, which would otherwise make optimization difficult. On that note, NeuralFoil will also give you an `"analysis_confidence"` output, which is a measure of uncertainty. Below, we show the same figure as before, but color the NeuralFoil results by analysis confidence. This illustrates how regions with delicate or uncertain aerodynamic behavior are flagged.

<p align="center">
	<img src="./benchmarking/neuralfoil_point_comparison_with_analysis_confidence.svg" width="1000" />
</p>

Due to domain knowledge embedded into its architecture, NeuralFoil is unusually capable of accurate generalization well beyond its training data. For example, the figure below shows that NeuralFoil can accurately predict aerodynamics on airfoils with extreme control surface deflections - despite the fact that none of NeuralFoil's training samples have deflected control surfaces. More details on this benchmark setup are available in the [NeuralFoil whitepaper](./paper/out/main.pdf).

<p align="center">
    <img src="./studies/control_surface_accuracy.svg" width="700" />
</p>

### Speed

In the table below, we quantify the performance of the NeuralFoil ("NF") models with respect to XFoil more precisely. At a basic level, we care about two things:

- **Accuracy**: how close are the predictions to XFoil's?
- **Computational Cost**: how long does it take to run?

This table details both of these considerations. The first few columns show the error with respect to XFoil on the test dataset. [The test dataset is completely isolated from the training dataset, and NeuralFoil was not allowed to learn from the test dataset](#geometry-parameterization-and-training-data). Thus, the performance on the test dataset gives a good idea of NeuralFoil's performance "in the wild". The second set of columns gives the runtime speed of the models, both for a single analysis and for a large batch analysis.

<table><thead><tr><th>Aerodynamics Model</th><th colspan="4">Mean Absolute Error (MAE) of Given Metric, on the Test Dataset, with respect to XFoil</th><th colspan="2">Computational Cost to Run (CPU)</th></tr></thead><tbody><tr><td></td><td>Lift Coeff.<br>$C_L$</td><td>Fractional Drag Coeff.<br>$\ln(C_D)$&nbsp;&nbsp;&nbsp;†</td><td>Moment Coeff.<br>$C_M$</td><td>Transition Locations<br>$x_{tr}/c$</td><td>Runtime<br>(1 run)</td><td>Total Runtime<br>(100,000 runs)</td></tr>
<tr><td>NF "xxsmall"</td><td>0.040</td><td>0.078</td><td>0.007</td><td>0.044</td><td>1.2 ms</td><td>0.87 sec</td></tr>
<tr><td>NF "xsmall"</td><td>0.030</td><td>0.057</td><td>0.005</td><td>0.033</td><td>1.2 ms</td><td>1.03 sec</td></tr>
<tr><td>NF "small"</td><td>0.027</td><td>0.050</td><td>0.005</td><td>0.027</td><td>1.3 ms</td><td>1.14 sec</td></tr>
<tr><td>NF "medium"</td><td>0.020</td><td>0.039</td><td>0.003</td><td>0.020</td><td>1.3 ms</td><td>1.36 sec</td></tr>
<tr><td>NF "large"</td><td>0.016</td><td>0.030</td><td>0.003</td><td>0.014</td><td>1.3 ms</td><td>2.34 sec</td></tr>
<tr><td>NF "xlarge"</td><td>0.013</td><td>0.024</td><td>0.002</td><td>0.010</td><td>1.4 ms</td><td>2.80 sec</td></tr>
<tr><td>NF "xxlarge"</td><td>0.012</td><td>0.022</td><td>0.002</td><td>0.009</td><td>1.6 ms</td><td>5.13 sec</td></tr>
<tr><td>NF "xxxlarge"</td><td>0.012</td><td>0.020</td><td>0.002</td><td>0.007</td><td>6.1 ms</td><td>12.0 sec</td></tr>
<tr><td>XFoil</td><td>0</td><td>0</td><td>0</td><td>0</td><td>73 ms</td><td>42 min</td></tr>
</tbody></table>

> † The deviation of $\ln(C_D)$ can be thought of as "the typical relative error in $C_D$". For example, if the mean absolute error (MAE) of $\ln(C_D)$ is 0.020, you can think of it as "typically, drag is accurate to within 2.0% of XFoil."

A better way to look at this tradeoff against XFoil is to assess speedup *while controlling for equivalent accuracy*. (After all, [it is usually trivial to get a speedup if you don't care about accuracy - just use a coarser discretization](https://x.com/shoyer/status/1362301955243057154).) This is shown in the plot below, where we vary the accuracy "knobs" for both XFoil and NeuralFoil - discretization resolution for XFoil, and model size for NeuralFoil. As shown here, NeuralFoil achieves a ~30x speedup over XFoil for a given level of accuracy, if a single analysis is run. For batched analyses, the vectorization advantage of NeuralFoil can result in speedups of nearly 1,000x at the same accuracy. More details on this benchmark setup are available in the [NeuralFoil whitepaper](./paper/out/main.pdf).

![Speed-accuracy trade against XFoil](./studies/speed_vs_xfoil_at_constant_accuracy/speed_vs_accuracy_tradeoff.svg)

Based on these performance numbers, you can select the right tradeoff between accuracy and computational cost for your application. In general, I recommend starting with the ["large"](#overview) model and adjusting from there.

In addition to accuracy vs. speed, another consideration when choosing the right model is what you're trying to use NeuralFoil for. Larger models will be more complicated ("less parsimonious," as the math kids would say), which means that they may have more "wiggles" in their outputs as they track XFoil's physics more closely. This might be undesirable for gradient-based optimization. On the other hand, larger models will be able to capture a wider range of airfoils (e.g., nonsensical, weirdly-shaped airfoils that might be seen mid-optimization), so larger models could have a benefit in that sense. If you try a specific application and have better/worse results with a specific model, let me know by opening a GitHub issue!

## Airfoil Shape Optimization using NeuralFoil

NeuralFoil can be used for airfoil shape optimization, in conjunction with [AeroSandbox](https://www.github.com/peterdsharpe/AeroSandbox). An example airfoil design optimization result is given in the [NeuralFoil whitepaper](./paper/out/main.pdf), with [code here](https://github.com/peterdsharpe/AeroSandbox/blob/master/tutorial/06%20-%20Aerodynamics/02%20-%20AeroSandbox%202D%20Aerodynamics%20Tools/02%20-%20NeuralFoil%20Optimization.ipynb). Here, we optimize an airfoil shape for a human-powered aircraft. This is a drag-minimization problem, subject to lift and pitching moment constraints, and manufacturing limits - full details in the paper. 

![daedalus_optimization.svg](./paper/TeX/figures/daedalus_optimization.svg)

Here, NeuralFoil achieves performance comparable to expert-designed airfoils. The entire optimization process takes roughly 30 seconds on a PC; optimization studies with a lower NeuralFoil `model_size` value can run as quick as half a second. Notably, if the problem formulation is well-posed, NeuralFoil will not "over-optimize" to achieve a solution that performs well at on-design conditions but very poorly when off-design. Compared to optimization by simple wrapping of XFoil with a gradient-based optimizer, the resulting airfoils achieve better aerodynamic performance due to the [ragged nature of XFoil's gradients](https://websites.umich.edu/~mdolaboratory/pdf/Adler2022c.pdf). And, compared to [wrapping XFoil with a gradient-free optimizer](https://github.com/jxjo/Xoptfoil2), NeuralFoil-based optimization is much faster.

## Extended Features (transonics, post-stall, control surface deflections)

For more sophisticated airfoil aerodynamics calculations, consider using NeuralFoil via [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox) (specifically, through [`asb.Airfoil.get_aero_from_neuralfoil()`](https://github.com/peterdsharpe/AeroSandbox/blob/8ad83aa4e4e40c503884c722143b7730c08089fa/aerosandbox/geometry/airfoil/airfoil.py#L607)). This provides several advanced features:

* **Compressible aerodynamics**, including transonic and supersonic aerodynamics. AeroSandbox will generally get the critical Mach number accurate to within $\pm 0.01$ or so. Subsonic corrections done using a Laitone correction (a higher-order variant of Prandtl-Glauert and Karman-Tsien). Wave drag accuracy is, of course, less reliable beyond the drag-divergence Mach number, although it still [agrees reasonably closely when compared to RANS CFD](https://github.com/peterdsharpe/AeroSandbox/blob/master/studies/WingTransonics/compare_methods.py).
* **Post-stall aerodynamics** (i.e., truly 360 degree range of $\alpha$). This is useful for applications like wind turbine blades or propeller roots, where the airfoil may be operating at high angles of attack.
* **Control surface deflections**. Currently only trailing-edge control surface deflections are supported in AeroSandbox's NeuralFoil interface.

Validation cases for all three features are given in the [NeuralFoil whitepaper](./paper/out/main.pdf).

## Installation

[Install from PyPI](https://pypi.org/project/NeuralFoil/) with `pip install neuralfoil`.

<a name="dependencies-question"></a>
To run models, NeuralFoil currently requires minimal dependencies:

* Python 3.7+
* [NumPy](https://numpy.org/)
* [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox) 4.0.10+

## Geometry Parameterization and Training Data

#### Geometry Parameterization

<a name="parameterization-question"></a>

As a user, you can give an airfoil in many different formats—for example, as a set of $(x,y)$ coordinates, as a .dat file, or as an AeroSandbox `Airfoil` object. However, under the hood, NeuralFoil parameterizes the airfoil geometry using the CST (Kulfan) parameterization. (You can also directly pass in Kulfan parameters if preferred.)

The airfoil shape fed into NeuralFoil's neural networks is in the form of an 8-parameter-per-side CST (Kulfan) parameterization, with Kulfan's added leading-edge-modification (LEM) and trailing-edge thickness parameter. This gives a total of (8 * 2 + 1 + 1) = 18 parameters to describe a given airfoil shape.

<p align="center">
    <img src="./media/kulfan_parameterization_illustration.svg" width="700" />
</p>

For more details on this parameterization, or why it is a good choice, read:

- [D. A. Masters, "Geometric Comparison of Aerofoil Shape Parameterization Methods", AIAA Journal, 2017.](https://arc.aiaa.org/doi/pdf/10.2514/1.J054943)
- The seminal paper on the CST (Kulfan) parameterization technique: [Brenda Kulfan, "Universal Parametric Geometry Representation Method"](https://www.brendakulfan.com/research)

To convert between airfoil coordinates and the CST parameterization, use the following functions:

```python
from aerosandbox.geometry.airfoil.airfoil_families import get_kulfan_parameters, get_kulfan_coordinates
```

with documentation [here](https://aerosandbox.readthedocs.io/en/master/autoapi/aerosandbox/geometry/airfoil/airfoil_families/index.html) or in the source ([here](https://github.com/peterdsharpe/AeroSandbox/blob/8ad83aa4e4e40c503884c722143b7730c08089fa/aerosandbox/geometry/airfoil/airfoil_families.py#L128), [here](https://github.com/peterdsharpe/AeroSandbox/blob/8ad83aa4e4e40c503884c722143b7730c08089fa/aerosandbox/geometry/airfoil/airfoil_families.py#L265)).

#### Training Data

To be written, but in the meantime [see here](https://github.com/peterdsharpe/NeuralFoil/tree/master/training) for details on the [synthetic data generation](https://github.com/peterdsharpe/NeuralFoil/tree/master/training/gen2_architecture/training_data) and [training processes](https://github.com/peterdsharpe/NeuralFoil/blob/master/training/gen2_architecture/train_blind_neural_network.py). Training data is not (yet) uploaded to GitHub, but will be soon - need to set up Git LFS, as it's many gigabytes. Contact me if you need it sooner.

Full details on the training data and test/train split are available in the [NeuralFoil whitepaper](./paper/out/main.pdf).

## FAQs

Will NeuralFoil be integrated directly into [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox)?

> [It already is](#extended-features-transonics-post-stall-control-surface-deflections)! In fact, NeuralFoil's advanced features are only available through its AeroSandbox interface ([demo](https://github.com/peterdsharpe/AeroSandbox/blob/master/tutorial/06%20-%20Aerodynamics/02%20-%20AeroSandbox%202D%20Aerodynamics%20Tools/01%20-%20NeuralFoil.ipynb)). However, the goal is to *also* keep this NeuralFoil repository available as a small stand-alone module, if desired. This simplifies dependencies for people using NeuralFoil in non-design applications (e.g., flight simulation, real-time control on embedded systems, etc.), and makes it easier if someone wants to port NeuralFoil to another language.

<a name="xfoil-benefit-question"></a>
Why not just use XFoil directly?

> XFoil is a truly excellent piece of aerospace software engineering and is the gold standard of airfoil analysis, for good reason. When its assumptions hold (airfoils in subsonic flow without massive separation), [**XFoil's accuracy actually exceeds that of RANS CFD**](https://www.sciencedirect.com/science/article/abs/pii/S1270963816300839), yet it has ~1000x lower computational cost. XFoil shines in particular for human-in-the-loop airfoil design. However, XFoil is not the right tool for all applications, for a few reasons:
> 
> - XFoil exhibits hysteresis: you can get slightly different solutions (for the same airfoil, $\alpha$, and $Re$) depending on whether you sweep $\alpha$ up or down, as Newton iteration is resumed from the last converged solution and uniqueness is not guaranteed. This hysteresis can be a big problem for design optimization.
> - XFoil is not differentiable, in the sense that it doesn't tell you how performance changes w.r.t. airfoil shape (via, for example, an adjoint). That's okay—NeuralFoil doesn't either, at least out-of-the-box. However, the "path to obtain an efficient gradient" is very straightforward for NeuralFoil's pure NumPy code, where many excellent options exist (e.g., [JAX](https://github.com/jax-ml/jax)). In contrast, gradient options for Fortran code (the language XFoil is in) either don't exist or are significantly less advanced (e.g., Tapenade). The most promising option for XFoil is probably [CMPLXFOIL](https://github.com/mdolab/CMPLXFOIL), which computes complex-step (effectively, forward-mode) gradients. However, even if you can get a gradient from XFoil, it still may not be very useful, because...
> - [XFoil's solutions intrinsically lack $C^1$-continuity](./studies/nf_vs_xf_smoothness_demo.svg). NeuralFoil, by contrast, is guaranteed to be $C^\infty$-continuous by construction. This is critical for gradient-based optimization.
>   - Even if one tries to compute gradients of XFoil's outputs by finite-differencing or complex-stepping, these gradients are often inaccurate.
>   - Notably, this is *not* just due to limited precision of XFoil's reported outputs - even at arbitrary floating point precision, its actual mathematical formulation is fundamentally not $C^1$-continuous.
>   - A bit into the weeds, but: this comes down to how XFoil handles transition (onset of turbulence). XFoil does a cut-cell approach on the transitioning interval, and while this specific cut-cell implementation restores $C^0$-continuity (i.e., transition won't truly "jump" from one node to another discretely), gradients of the laminar and turbulent BL closure functions still change at the cell interface due to the differing BL parameters ($H$ and $Re_\theta$) from node to node. This loses $C^1$ continuity, causing a "ragged" polar at the microscopic level. In theory $C^1$-continuity could be restored by [also blending the BL shape variables through the transitioning cell interval](https://dspace.mit.edu/handle/1721.1/119272) (intermittency), but that unleashes some ugly integrals and is not done in XFoil.
>     - For more on this, see [Adler, Gray, and Martins, "To CFD or not to CFD?..."](http://websites.umich.edu/~mdolaboratory/pdf/Adler2022c.pdf), Figure 7.
> - While XFoil is ~1000x faster than RANS CFD, NeuralFoil [can be another ~1000x faster to evaluate than XFoil](#performance). NeuralFoil is also much easier to interface with on a memory level than XFoil, which means you won't find yourself I/O bound from file reading/writing like you will with XFoil. ([Memory interfacing with XFoil is possible](https://github.com/DARcorporation/xfoil-python), but rare.)
> - XFoil is not vectorized, which exacerbates the speed advantage of a (vectorized) neural network when analyzing large batches of airfoil cases simultaneously.
> - XFoil is not guaranteed to produce a solution, and often crashes when "ambitious" calculations are attempted. By contrast, NeuralFoil will always produce *an* answer, even if less accurate (and in these cases, the "analysis_confidence" metric will warn you that these results are dubious). In some applications, XFoil's occasional non-convergence is okay or even desirable; in others, that's a deal-breaker. Example applications where this is a problem include:
>   - Real-time control, where one wants to estimate forces (e.g., for a MPC trajectory), but you can't have the controller crash if XFoil fails to converge or hangs the CPU.
>   - Flight simulation: similar to real-time control where "a less-accurate answer" is much better than "no answer."
>   - Design optimization, where the optimizer needs "an answer" in order to recover from a bad design point and send the search back to a reasonable design.
> - XFoil can be a serious pain to compile from source, which is often required if running on Mac or Linux (i.e., all supercomputers, some lab computers). NeuralFoil is pure Python and NumPy, so it's easy to install and run anywhere.

Why not use a neural network trained on RANS CFD instead?

> This is a cool idea too, and it has been done (See [Bouhlel, He, and Martins, "Scalable gradient-enhanced artificial..."](https://link.springer.com/article/10.1007/s00158-020-02488-5))! The fundamental challenge here, of course, is the cost of training data. RANS CFD is much more expensive than XFoil, so it's much harder to get sufficient training data to build a neural network that will generalize well out-of-sample. For example, in the linked work by Bouhlel et al., the authors trained a neural network on 42,000 RANS CFD runs (and they were sweeping over Mach as well, so the data becomes even sparser). In contrast, NeuralFoil was trained on tens of millions of XFoil runs. Ultimately, this exposes NeuralFoil to a much larger "span" of the airfoil design space, which is critical for accurate predictions on out-of-sample airfoils.
> 
> One advantage of a RANS CFD approach over the NeuralFoil XFoil approach is, of course, transonic modeling. NeuralFoil attempts to get around this a little bit by estimating $C_{p, min}$, which in turn allows you to estimate the critical Mach number. (For an implementation of that, [see here](#extended-features-transonics-post-stall-control-surface-deflections)) But fundamentally, NeuralFoil is likely less accurate in the transonic range because of this. The tradeoff is that the much larger training data set allows NeuralFoil to be more accurate in the subsonic range, where [XFoil is actually usually more accurate than RANS CFD](https://www.sciencedirect.com/science/article/abs/pii/S1270963816300839).

What's the underlying neural network architecture used in NeuralFoil? In what sense is it "physics-informed"?

> Surprisingly basic - when all the peripherals are stripped away, the learned core itself is a simple MLP with a varying number of total layers and layer width depending on model size. Layer counts and widths were [determined through extensive trial and error](./training/supercloud_job_id_notes.log), in conjunction with observed test- and train-loss values. All layers are dense (fully connected, with weights and biases). All activation functions between layers are $\tanh$, to preserve $C^\infty$-continuity. The number of layers and layer width are as follows:
> 
> * xxsmall: 2 layers,  32 wide.
> * xsmall:  3 layers,  32 wide.
> * small:   3 layers,  48 wide.
> * medium:  4 layers,  64 wide.
> * large:   4 layers, 128 wide.
> * xlarge:  4 layers, 256 wide.
> * xxlarge: 5 layers, 256 wide.
> * xxxlarge:5 layers, 512 wide.
>
> The domain knowledge embedding (the "physics-informed" part) happens primarily in a) encoding/decoding latent space choices, b) symmetry embedding, and c) how the model dynamically fuses a learned model and an empirical model, depending on the uncertainty of the learned model. NeuralFoil is "physics-informed", but notably not a [PINN](https://en.wikipedia.org/wiki/Physics-informed_neural_networks). ([To dispel a common misconception, "physics informed machine learning" is an umbrella term that extends far beyond just PINNs - see Steve Brunton's taxonomy here](https://youtu.be/JoFW2uSd3Uo).) NeuralFoil is an interesting case study about how full-field learning using sophisticated ML architectures (e.g., PINNs, neural operators, CNNs/GNNs) is not always the only or best way to embed physics domain knowledge into a model. In fact, simple strategies can often yield compelling tradeoffs, as measured by speed, accuracy, data efficiency, and generalizability.
>
> Spiritually, NeuralFoil's performance is perhaps ~75% attributable to classical fluid dynamics knowledge embedded into the architecture, and 25% due to the learned core.

Could you make NeuralFoil more accurate, relative to XFoil, by a) increasing the shape parameterization dimensionality or b) increasing the neural network size?

> Yes. *But you probably don't actually want to make it more accurate*—and not for any of the reasons that ML engineers would typically cite (slower runtime or increased data requirements). This may sound puzzling, so let's unpack this.
>
> NeuralFoil's true goal is to serve as a *useful tool* for practical aerospace engineers to design airfoils that physically work well in real-world applications—you should be able to take a NeuralFoil airfoil, actually go fly it, and get similar performance. Critically, this goal is different than just accuracy alone. As stated by Mark Drela (MIT; author of XFoil) in personal correspondence about NeuralFoil:
> 
> > The reason why the NeuralFoil-optimized airfoil does not exhibit the 
point-optimization problem seen in the Pros & Cons paper is likely because 
of its smaller geometry design space. The small separation-bubble sized bumps
which cause the problem cannot appear in this space. **In practice, this is 
likely a nice feature and not a bug.**  [You] might want to mention this because 
it's not obvious.
> 
> On its highest-offered accuracy setting ("xxxlarge"), NeuralFoil's error in drag prediction  against XFoil is currently roughly 0.38%. A second reason not to try to increase accuracy further is that, although XFoil is quite accurate, 0.38\% drag error is well below the "noise floor" of what XFoil can reliably capture, due to modeling assumptions. So, further decreasing NeuralFoil's surrogate modeling error below 0.38\% may make for attractive metrics, but the actual accuracy with respect to reality (and hence, the *practical* utility) will not meaningfully improve.
> 
> A third reason is that, as you increase the neural network size, you increase the risk of overfitting to XFoil's inherent non-smoothness. We specifically do not want this, as the airfoil design optimization case study shows that XFoil's non-smoothness can be a significant problem for gradient-based optimization. Limiting the parameter count essentially low-pass filters the learned mapping, which is desirable. 

## Acknowledgements

NeuralFoil was trained on [MIT Supercloud](https://supercloud.mit.edu/), a high-performance computing cluster operated by the MIT Lincoln Laboratory Supercomputing Center (LLSC).

## License

NeuralFoil is licensed under [the MIT license](LICENSE.txt).

## Citing NeuralFoil

If you use NeuralFoil in your research, please cite:

Both the tool itself (this repository), which includes the [pre-print publication](./paper/out/main.pdf):

```bibtex
@misc{neuralfoil,
  author = {Peter Sharpe},
  title = {{NeuralFoil}: An airfoil aerodynamics analysis tool using physics-informed machine learning},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/peterdsharpe/NeuralFoil}},
}
```

And [the author's PhD thesis](https://dspace.mit.edu/handle/1721.1/157809), which has an extended chapter that serves as the primary long-form documentation for the tool:

```bibtex
@phdthesis{aerosandbox_phd_thesis,
   title = {Accelerating Practical Engineering Design Optimization with Computational Graph Transformations},
   author = {Sharpe, Peter D.},
   school = {Massachusetts Institute of Technology}, 
   year = {2024},
}
```
