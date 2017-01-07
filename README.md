# standing-travelling-decomp

Introduction
------------

The routines included in this repository implement the spectral decomposition defined in Watt-Meyer and Kushner (JAS, 2015). This decomposition takes in two-dimensional data (usually time versus longitude), applies a 2D discrete Fourier transform, and then further separates the Fourier coefficients into standing and travelling components. The details of the method are described in the above-referenced article.


Description of contents
-----------------------

This repository includes the following files:

**README.md:** this README

**stand_travel_example_reanalysis.py:** a python script that inputs reanalysis data (the ERA-Interim daily-mean 500hPa geopotential for the period 1 November 1979 to 30 March 1980), applies the spectral decomposition to it, and makes some representative plots.

**stand_travel_example_reanalysis.py:** a python script that generates artifical time- and longitude-dependent data, applies the spectral decomposition to it, and makes some representative plots.

**stand_travel_routines.py:** a python file which includes the routines necessary to implement the standing-travelling decomposition, as well as make some relevant plots.

**z_anom_500hPa_NDJFM_1979-1980_ERAInterim.nc:** netCDF file of the ERA-Interim daily-mean 500hPa geopotential for the period 1 November 1979 to 30 March 1980, for use in **stand_travel_example_reanalysis.py**.


References
----------

Watt-Meyer, O., and P. J. Kushner (2015), Decomposition of atmospheric disturbances into standing and traveling components, with application to Northern Hemisphere planetary waves and stratosphere-troposphere coupling. J. Atmos. Sci., **72,** 787-802.
