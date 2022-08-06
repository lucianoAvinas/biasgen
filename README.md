# biasgen

**biasgen** is a Python package which simulates magnetic resonance (MR) non-uniformities. This package allows users to generate custom bias fields by providing radiofrequency (RF) coil spatial information as well as frequency sampling information.

Bias fields are constructed from a sum-of-squares approach using sensitivity maps which follow the sinusoidal model of [Kern et. al](https://ieeexplore.ieee.org/document/6062681). Direct computation of sinusoidal sensitivities are done using the closed form solution of segment sources derived by [Vinas and Sudhyadhom](https://arxiv.org).

# Installation

Package **biasgen** can be installed using pip:
```
pip install biasgen
```
A CUDA accelerated version of **biasgen** is also provided and can be installed with:
```
pip install biasgen[gpu]
```

# Usage

**biasgen** requires a set of coil positions and sampling frequencies in order to work. Coil positioning can be initialized using several  *CoilSegment* objects or through the predefined constructor routine *cage_constructor*. An example using *cage_constructor* is provided below:
```python
import biasgen

ph = biasgen.phantom3d(n=128)
coils = biasgen.cage_constructor(n_coils=2, center=(64,64,64), coil_height=128,
                                 length_to_space_ratio=0.35, ellip_axes=(90,65))
biasgen.view_coil_positioning(ph, coils)
```
<p align="center">
  <img width="300" height="300" src="images/2coil_example.png">
</p>

Here function *view_coil_positioning* provides a top-down view of the coils defined by our cage constructor. Next is to define a sampling grid for the sinusoidal sensitivity model:
```python
sens_settings = biasgen.SensitivitySettings(grid_lengths=(5,5,5), grid_spacings=(1,1,1))
```

These sampling settings can be used in conjunction to the coils to produce a set of sensitivity maps:
```python
# biasgen.use_gpu(True) # Uncomment if GPU is available
sens = biasgen.compute_sensitivity(coils, sens_settings, ph.shape, batch_sz=1, scale_fctr=0.5)
```

Arguments "batch_sz" and "scale_fctr" can be helpful for memory-limited devices. "batch_sz" determines how many segments are loaded into memory while "scale_fctr" provides a temporary spatial downsampling during computation. The sensitivity maps can be viewed using function *view_center_axes* and a boolean mask:
```python
ph_mask = ph > 0
biasgen.view_center_axes(abs(sens), ph_mask, ['Z-slice','Y-slice','X-slice'])
```
<p align="center">
  <img width="516" height="388" src="images/2coil_sens_maps.png">
</p>

Finally the bias field is constructed through a sum-of-squares procedure:
```python
bias = biasgen.bias_sum_of_squares(sens)
biasgen.view_center_axes(bias*ph, ph_mask, ['Z-slice','Y-slice','X-slice'])
```
<p align="center">
  <img width="516" height="186" src="images/2coil_biased_phantom.png">
</p>

More detailed examples can be found in the examples/bias.ipynb notebook.

# References
1. Guerquin-Kern M, Lejeune L, Pruessmann KP, Unser M. Realistic
Analytical Phantoms for Parallel Magnetic Resonance Imaging. IEEE
Transactions on Medical Imaging. 2012;31(3):626-636. 
2. TBD