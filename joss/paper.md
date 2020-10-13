---
title: 'climpred: Verification of weather and climate forecasts'
tags:
  - Python
  - climate
  - forecasting
authors:
  - name: Riley X. Brady
    orcid: 0000-0002-2309-8245
    affiliation: 1
  - name: Aaron Spring
    orcid: 0000-0003-0216-2241
    affiliation: 2
affiliations:
 - name: Department of Atmospheric and Oceanic Sciences and Institute of Arctic and Alpine Research, University of Colorado Boulder, Boulder, Colorado USA
   index: 1
 - name: Max Planck Institute for Meteorology, Hamburg, Germany
   index: 2
date: 13 October 2020
bibliography: paper.bib
---

# Summary

Predicting extreme events and variations in weather and climate yields numerous benefits
for economic, social, and environmental decision-making. However, quantifying prediction
skill for multi-dimensional geospatial model output is computationally expensive and a
difficult coding challenge. The large datasets (order gigabytes to terabytes) require
parallel and out-of-memory computing to be analyzed efficiently. Further, aligning the
many forecast initialization and target dates with differing observational products is a
straight-forward, but exhausting and error-prone exercise for researchers.

To simplify and standardize forecast verification across scales from hourly weather to
decadal climate forecasts, we built `climpred`: a community-driven python package for
computationally efficient and methodologically consistent verification of ensemble
prediction models. The code base is currently maintained through open-source
development. It leverages `xarray` [@Hoyer:2017] to anticipate core prediction ensemble
dimensions (_e.g._, ensemble member as `member`, initialization date as `init`, and lead
time as `lead`) and `dask` @Rocklin:2015 to perform out-of-memory and parallelized
computations on large datasets.

`climpred` aims to offer a comprehensive set of analysis tools for assessing the quality
of dynamical forecasts relative to verification products (_e.g._, observations,
reanalysis products, control simulations). The package includes a suite of deterministic
and probabilistic verification metrics that are constantly expanded by the community and
are generally organized in the open-source package `xskillscore`. To avoid casting too
wide of a scope, `climpred` expects users to handle their domain- and
institutional-specific post-processing of raw model output, so that the package can
focus on the actual analysis component of weather and climate forecasting.

# Statement of Need
While similar verification packages have already been released (_e.g._,
`s2dverification` [@Manubens:2018] written in R and `MurCSS` [@Illing:2014] written with
python-based `CDO`-bindings [@CDO]), `climpred` is unique for many reasons. (1) It
spans all temporal scales of prediction, including weather, subseasonal-to-seasonal (S2S),
and seasonal-to-decadal (S2D). (2) `climpred` works across all computational scales,
from personal laptops to supercomputers (HPC), and in the cloud. This is mostly due to
its support of out-of-memory computation with `dask` [@dask]. (3) `climpred` is highly
modular and supports the research process from end-to-end, from loading in model output,
to interactive pre-processing and analysis, to vizualization in python. (4) `climpred`
is part of and benefits from the wider scientific python community, `pangeo`. A wide
adoption of `climpred` could standardize prediction model evaluation and make
verification reproducible [@Irving2015]. (5) The `climpred` documentation will serve as
a repository of unified analysis methods through `jupyter` notebook [@Kluyver:2016]
examples and will also collect relevant references and literature.

# Prediction Simulation Types
Weather and climate modeling institutions typically run so-called "hindcasts," where
they retrospectively initialize dynamical models from many past observed climate states
[@Meehl:2009]. Initializations are then slightly perturbed to generate an ensemble of
forecasts that diverge due to their sensitive dependence on initial conditions
[@Lorenz:1969]. Hindcasts are evaluated by using some statistical metric to score their
performance against historical observations. "Skill" is established by comparing these
results to the performance of some "reference" forecast (_e.g._, a persistence or
climatology forecast). The main assumption is that the skill established relative to the
past will propagate to forecasts of the future.

A more idealized approach is the so-called "perfect model" framework, which is ideal for
investigating processes leading to potentially exploitable predictability
[@Griffies:1997; @Bushuk:2018; @Seferian:2018; @Spring:2020]. Ensemble members are spun off
an individual model (by slightly perturbing its state) to predict its own evolution.
This avoids initialization shocks [@Kroger:2017], since the framework is self-contained.
However, it cannot predict the real world. The perfect model setup rather estimates the
theoretical upper limit timescale after which the value of dynamical initialization is
lost due to chaos in the system, assuming that the model perfectly replicates the
dynamics of the real world. Of course, this evaluation is subject to the structural
biases of the model used. Skill quantification is accomplished by considering one
ensemble member as the verification data and the remaining members as the forecasts
[@Griffies:1997].

`climpred` supports both of these formats, offering `HindcastEnsemble` and
`PerfectModelEnsemble` objects. `HindcastEnsemble` is instantiated with an
`initialized` hindcast ensemble dataset and requires an `observation` dataset against
which to verify. `PerfectModelEnsemble` is instantiated with an `initialized` perfect
model ensemble dataset and also accepts a `control` dataset against which to evaluate
forecasts. Both objects can also track an `uninitialized` dataset, which represents a
forecast that evolves solely due to random internal climate variability
[e.g., @Kay:2014]. "Uninitialized" is a bit of a misnomer, but is used in the decadal
climate prediction community [@Yeager:2018]. This means that the model simulation is
initialized once during model spinup, but not re-initialized. Once instantiated, the
objects can be manipulated by calling any `xarray` methods, by mapping other methods to
all contained datasets (such as `numpy` methods), through plotting, or by using
arithmetic operations.

# Forecast Verification
Assessing skill for `PredictionEnsemble` objects (the parent class to
`HindcastEnsemble` and `PerfectModelEnsemble`) is standardized into a one-liner:

```python
PredictionEnsemble.verify(
  metric='pearson_r',
  comparison='e2o',
  alignment='same_verif',
  dim='init',
  reference='persistence'
)
```

where `metric=...` identifies which statistical metric to use to score the forecast
(in this case the anomaly correlation coefficient from `xskillscore`), `comparison=...`
identifies how to compare the forecast ensemble to the observations (in this case the
ensemble mean against observations), `alignment=...` identifies how to subset the
initializations when verifying against the observations (in this case using the same set
of verification dates across all leads), `reference=...` identifies which reference
forecasts to include when verifying against observations (in this case a persistence
forecast, which simply persists the anomalies at initialization forward to all leads),
and `dim=...` indicates which dimension to reduce the operation over (in this case the
initialization dimension).

`climpred` draws from a metric library of 17 contingency table-derived,
14 deterministic, and 8 probabilistic metrics housed at our companion project
`xskillscore`. Users can add additional metrics through pull requests or dynamically
without touching the source code.

For `HindcastEnsemble` objects, users can choose from the following comparisons:

* `'e2o'`: Compare the ensemble mean to the observations.
* `'m2o'`: Compare each ensemble member individually to the observations.

For `PerfectModelEnsemble` objects, users can choose from the following comparisons:

* `'m2e'`: Compare all ensemble members to the ensemble mean while leaving out the
  verification member in the ensemble mean.
* `'m2c'`: Compare all member forecasts to a single verification member.
* `'m2m'`: Compare all members to all others in turn while leaving out the verification
  member.
* `'e2c'`: Compare the ensemble mean forecast to a single verification member.

Currently, the available reference forecasts to compare the initialized forecast to
are `'persistence'` and `'uninitialized'`. The former computes the results for a
forecast that simply persists observational anomalies forward to each lead time. The
latter evaluates the performance of a climate simulation that is only initialized a
single time and thus represents the skill due to internal variability or external
forcing (see @Yeager:2018 and @Brady:2020).

# Datetime Alignment
Hindcast systems initialize an ensemble of M members over N initializations for L lead
time steps (_e.g._, lead years). This makes for complicated datetime coordinates, where
each initialization has a time series corresponding to some "target" dates to align with
the verification data. In the following, we outline the three different alignment
methods found in the literature and through personal communications with modeling
groups (\autoref{Fig:1}):

1. "maximize" (`'maximize'`): Use all available initializations at each lead that verify
   against the observations provided. This changes both the set of initializations and
   the verification window used at each lead. This is particularly useful for short
   observational records and is used by, _e.g._, @Yeager:2018 and @Brady:2020.
2. "same initializations" (`'same_init'`): Use a common set of initializations that
   verify across all leads. This ensures that there is no bias in the result due to the
   state of the system for the given initializations. This is used by design in perfect
   model simulations and by the MurCSS plugin of the MiKlip Central Evaluation System
   [@Illing:2014].
3. "same verification" (`'same_verif'`): Select initializations based on a common
   verification window across all leads. This ensures that there is no bias in the
   result due to the observational period being verified against. This is typically used
   by the subseasonal-to-seasonal (S2S) community [@Hawkins:2014; @Pegion:2019].

The python built-in `logging` on level `INFO` can be used to track the initialization
and verification dates used for each lead.

![Schematic of alignment strategies for hindcast prediction systems. In all three subplots, the top panel shows the initialized forecast system with initialization years on the bottom row and lead years in the upper three rows. The bottom panel shows a sample verification dataset, spanning 1995 to 2002. (a) The "maximize" strategy, which maximizes degrees of freedom by selecting all initializations that verify with the given observational product at each lead. (b) The "same initialization" strategy, which selects a common set of initializations that can verify at all leads. (c) The "same verification" strategy, which selects a common set of verification dates at all leads. Note that the stipulation of having a union with observations (the vertical black bars) is only applied when a persistence reference forecast is selected, since the persistence forecast should use a set of initializations that are identical with the forecasting system for consistency.\label{Fig:1}](figures/Fig1.png)

# Significance Testing
Significance testing is important for assessing whether a given initialized prediction
system is skillful, _i.e._, statistically significant over some reference forecast.
`climpred` implements different approaches for significance testing:

1. parametric approach: For all correlation metrics, like the Pearson's or Spearmean's
   anomaly correlation coefficient, `climpred` returns an associated p value. The p
   value communicates the probability that the correlation found between the forecasts
   and observations would arise, assuming the two time series are uncorrelated and based
   on Gaussian distributions.
2. non-parametric approach: Testing statistical significance through bootstrapping with
   replacement is commonly used in the field of climate prediction
   [@Goddard:2013, @Boer:2016]. Bootstrapping relies on resampling the underlying data
   with replacement for a large number of iterations, as proposed by the decadal
   prediction framework of @Goddard:2013. In practice, the initialized ensemble is
   resampled with replacement along a dimension (generally the `member` or `init`
   dimension) and then that resampled ensemble is verified against the observations
   (over `N` bootstrapping iterations requested by the user). Resampling `init`
   approximates the sensitivity of the statistical result to the initializations that
   were sampled by the forecasting system, whereas resampling `member` approximates the
   sensitivity of the result to the number and characteristics of the ensemble members
   sampled. This resampling leads to a distribution of initialized skill values based on
   the resampled iterations of the ensemble. The resampling is also applied to the
   reference forecasts. The probability or p value is derived from the fraction of these
   resampled initialized metrics beaten by the resampled reference metrics calculated
   from their respective distributions. Confidence intervals using these distributions
   are also calculated.
3. sign test: @DelSole:2016 proposes a test that determines whether one forecast is
   significantly better than another forecast. It is derived from the statistics of a
   random walk. The sign test can be applied to a wide class of measures of forecast
   quality, including ordered (ranked) categorical data. It is also independent of
   distributional assumptions about the forecast errors. `climpred` offers examples of
   how to use the sign test function from `xskillscore` to compare forecast scores
   derived from `climpred` verification functions.

# Additional Functionality
For a smoother user experience, `climpred` offers a few utility functions:

* Decadal Climate Prediction Project (DCPP) output [@Boer:2016] from the Coupled Model
   Intercomparison Project Phase 6 (CMIP6) [@Eyring:2016] can be easily loaded and
   converted into `xarray` using `intake-esm`.
* A `PredictionEnsemble.smooth(...)` method for spatial and temporal smoothing,
  following [@Goddard:2013].
* A `HindcastEnsemble.remove_bias(...)` method to remove the mean bias between the
  initialized \texttt{HindcastEnsemble} and observations prior to evaluating its
  performance.
* A graphics library with quick plotting functions to view a prediction system.

# Acknowledgements

We thank Andrew Huang for early stage refactoring and continued feedback on `climpred`.
We also thank Kathy Pegion for pioneering the seasonal, monthly, and subseasonal time
resolutions. Thanks in addition to Ray Bell for initiating and maintaining
`xskillscore`, which serves to host the majority of metrics used in `climpred`.

# References
