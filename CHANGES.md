# Changelog History

## climpred v0.3 (2019-04-27)

### Features

-   Introduces object-oriented system to `climpred`, with classes `ReferenceEnsemble` and `PerfectModelEnsemble`. (#86) [`Riley Brady`]
-   Expands bootstrapping module for perfect-model configuartions. (#78, #87) [`Aaron Spring`]
-   Adds functions for computing Relative Entropy (#73) [`Aaron Spring`]
-   Sets more intelligible dimension expectations for `climpred`: (#98, #105) [Riley Brady; Aaron Spring]
    -   `init`:  initialization dates for the prediction ensemble
    -   `lead`:  retrospective forecasts from prediction ensemble; returned dimension for prediction calculations
    -   `time`:  time dimension for control runs, references, etc.
    -   `member`:  ensemble member dimension.

### Bug Fixes

-   `xr_rm_poly` can now operate on Datasets and with multiple variables. It also interpolates across NaNs in time series. (#94) [Andrew Huang]
-   Travis CI, `treon`, and `pytest` all run for automated testing of new features. (#98, #105, #106) [`Riley Brady`; `Aaron Spring`]

## climpred v0.2 (2019-01-11)

Name changed to `climpred`, developed enough for basic decadal prediction tasks on a perfect-model ensemble and reference-based ensemble.

## climpred v0.1 (2018-12-20)

Development begins, reworking the original `esmtools` package.
