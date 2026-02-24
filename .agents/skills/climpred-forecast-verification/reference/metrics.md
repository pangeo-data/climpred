# climpred Metrics Reference

## Contents
- [Deterministic Correlation Metrics](#deterministic-correlation-metrics)
- [Deterministic Distance Metrics](#deterministic-distance-metrics)
- [Normalized Distance Metrics](#normalized-distance-metrics)
- [Bias Decomposition Metrics](#bias-decomposition-metrics)
- [Probabilistic Metrics](#probabilistic-metrics)
- [Contingency Metrics](#contingency-metrics)
- [Metric Selection Guide](#metric-selection-guide)

## Deterministic Correlation Metrics

| Metric | Aliases | Range | Perfect | Description |
|--------|---------|-------|---------|-------------|
| `pearson_r` | `acc`, `corr` | [-1, 1] | 1 | Anomaly correlation coefficient |
| `pearson_r_p_value` | `p`, `p_val` | [0, 1] | 0 | p-value for Pearson r |
| `pearson_r_eff_p_value` | `p_eff` | [0, 1] | 0 | p-value with effective sample size |
| `effective_sample_size` | `n_eff` | [0, ∞] | — | Effective sample size (autocorrelation-adjusted) |
| `spearman_r` | `sacc` | [-1, 1] | 1 | Spearman rank correlation |
| `spearman_r_p_value` | `sp` | [0, 1] | 0 | p-value for Spearman r |
| `spearman_r_eff_p_value` | `sp_eff` | [0, 1] | 0 | p-value with effective sample size |

## Deterministic Distance Metrics

| Metric | Aliases | Range | Perfect | Description |
|--------|---------|-------|---------|-------------|
| `mse` | — | [0, ∞] | 0 | Mean squared error |
| `rmse` | — | [0, ∞] | 0 | Root mean squared error |
| `mae` | — | [0, ∞] | 0 | Mean absolute error |
| `me` | `bias` | (-∞, ∞) | 0 | Mean error (bias) |
| `median_absolute_error` | — | [0, ∞] | 0 | Median absolute error |
| `mape` | — | [0, ∞] | 0 | Mean absolute percentage error |
| `smape` | — | [0, 200] | 0 | Symmetric mean absolute percentage error |

## Normalized Distance Metrics

These are normalized by verification variance. Perfect score = 0 (or 1 for MSESS).

| Metric | Aliases | Range | Perfect | Description |
|--------|---------|-------|---------|-------------|
| `nmse` | `nev` | [0, ∞] | 0 | Normalized MSE (1 = no skill) |
| `nmae` | — | [0, ∞] | 0 | Normalized MAE |
| `nrmse` | — | [0, ∞] | 0 | Normalized RMSE (1 = no skill) |
| `msess` | `ppp`, `msss` | (-∞, 1] | 1 | MSE skill score (1 - NMSE) |

## Bias Decomposition Metrics

Murphy (1988) decomposition of MSE into reliability, resolution, and uncertainty.

| Metric | Aliases | Range | Perfect | Description |
|--------|---------|-------|---------|-------------|
| `unconditional_bias` | `u_b`, `additive_bias` | (-∞, ∞) | 0 | Unconditional (additive) bias |
| `mul_bias` | `m_b`, `multiplicative_bias` | [0, ∞] | 1 | Multiplicative bias (std ratio × r) |
| `conditional_bias` | `c_b`, `cond_bias` | [0, ∞] | 1 | Conditional bias |
| `bias_slope` | — | (-∞, ∞) | 1 | Slope of forecast vs observed regression |
| `uacc` | — | (-∞, 1] | 1 | Unbiased ACC |
| `std_ratio` | — | [0, ∞] | 1 | Standard deviation ratio (fcst/obs) |
| `msess_murphy` | `msss_murphy` | (-∞, 1] | 1 | MSESS via Murphy decomposition |
| `spread` | — | [0, ∞] | — | Ensemble spread (std over member) |

## Probabilistic Metrics

**Require `comparison="m2o"` (HindcastEnsemble) or `"m2m"/"m2c"` (PerfectModelEnsemble).**
**`dim` must include `"member"`.**

| Metric | Aliases | Range | Perfect | Description |
|--------|---------|-------|---------|-------------|
| `crps` | — | [0, ∞] | 0 | Continuous Ranked Probability Score |
| `crpss` | — | (-∞, 1] | 1 | CRPS skill score (vs climatology) |
| `crpss_es` | — | (-∞, 1] | 1 | CRPS skill score (vs Gaussian spread) |
| `brier_score` | `bs` | [0, 1] | 0 | Brier Score (requires `category_edges`) |
| `threshold_brier_score` | `tbs` | [0, 1] | 0 | Brier Score at thresholds |
| `rps` | — | [0, 1] | 0 | Ranked Probability Score (requires `category_edges`) |
| `discrimination` | — | — | — | Discrimination histogram |
| `reliability` | — | — | — | Reliability diagram values |
| `rank_histogram` | — | — | uniform | Rank histogram (Talagrand diagram) |
| `roc` | — | [0, 1] | 1 | Receiver Operating Characteristic area |
| `less` | — | — | — | Logarithmic Ensemble Spread Score |

### Probabilistic metric kwargs

**`brier_score`** and **`rps`** require `category_edges`:
```python
hindcast.verify(
    metric="rps",
    comparison="m2o",
    dim=["init", "member"],
    alignment="same_inits",
    category_edges=np.array([0, 0.5, 1.0]),  # bin edges
)
```

**`threshold_brier_score`** requires `threshold`:
```python
hindcast.verify(
    metric="threshold_brier_score",
    comparison="m2o",
    dim=["init", "member"],
    alignment="same_inits",
    threshold=0.5,
)
```

## Contingency Metrics

Use `metric="contingency"` with `score` kwarg:

```python
hindcast.verify(
    metric="contingency",
    comparison="m2o",
    dim=["init", "member"],
    alignment="same_inits",
    score="accuracy",  # or any xskillscore.Contingency method
    observation_category_edges=np.array([-np.inf, 0, np.inf]),
    forecast_category_edges=np.array([-np.inf, 0, np.inf]),
)
```

Available `score` values: `"accuracy"`, `"bias_score"`, `"hit_rate"`, `"false_alarm_rate"`, `"false_alarm_ratio"`, `"odds_ratio"`, `"odds_ratio_skill_score"`, `"heidke_score"`, `"peirce_score"`, `"gerrity_score"`

## Metric Selection Guide

| Goal | Recommended Metric | Comparison |
|------|-------------------|------------|
| Correlation skill | `acc` (pearson_r) | `e2o` |
| Error magnitude | `rmse` or `mae` | `e2o` |
| Bias | `me` | `e2o` |
| Normalized skill (comparable across vars) | `msess` or `nrmse` | `e2o` |
| Ensemble reliability | `crps` | `m2o` |
| Categorical skill | `brier_score` or `rps` | `m2o` |
| Ensemble spread-skill | `spread` vs `rmse` | `m2o` / `e2o` |
| Statistical significance | `pearson_r_eff_p_value` | `e2o` |
