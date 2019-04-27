# Changelog History

## climpred v0.3 (2019-04-27)

### Features
* Introduces object-oriented system to `climpred`, with classes `ReferenceEnsemble` and `PerfectModelEnsemble`. (#86) [Riley Brady]
* Expands bootstrapping module for perfect-model configuartions. (#78, #87) [Aaron Spring]
* Adds functions for computing Relative Entropy (#73) [Aaron Spring]

### Bug Fixes
* `xr_rm_poly` can now operate on Datasets and with multiple variables. It also interpolates across NaNs in time series. (#94) [Andrew Huang]

## climpred v0.2 (2019-01-11)
Name changed to `climpred`, developed enough for basic decadal prediction tasks on a perfect-model ensemble and reference-based ensemble.

## climpred v0.1 (2018-12-20)
Development begins, reworking the original `esmtools` package.
