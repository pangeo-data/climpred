# climpred Comparisons Reference

## Contents
- [HindcastEnsemble Comparisons](#hindcastensemble-comparisons)
- [PerfectModelEnsemble Comparisons](#perfectmodelensemble-comparisons)
- [Choosing a Comparison](#choosing-a-comparison)

## HindcastEnsemble Comparisons

| Comparison | Aliases | Type | Description |
|-----------|---------|------|-------------|
| `e2o` | `e2r` | Deterministic only | Ensemble mean vs observations |
| `m2o` | `m2r` | Deterministic + Probabilistic | Each member vs observations |

### e2o (ensemble mean vs observations)
- Computes ensemble mean first, then compares to observations
- Use for deterministic metrics (`acc`, `rmse`, `mae`, `msess`, etc.)
- `dim` should NOT include `"member"`

```python
hindcast.verify(metric="rmse", comparison="e2o", dim="init", alignment="same_inits")
```

### m2o (members vs observations)
- Compares each member individually to observations
- Required for probabilistic metrics (`crps`, `rps`, `brier_score`, etc.)
- `dim` must include `"member"` for probabilistic metrics

```python
hindcast.verify(
    metric="crps", comparison="m2o", dim=["init", "member"], alignment="same_inits"
)
```

## PerfectModelEnsemble Comparisons

| Comparison | Type | Description |
|-----------|------|-------------|
| `m2m` | Deterministic + Probabilistic | All members vs all other members (leave-one-out) |
| `m2e` | Deterministic + Probabilistic | Each member vs ensemble mean |
| `m2c` | Deterministic + Probabilistic | Each member vs control run |
| `e2c` | Deterministic only | Ensemble mean vs control run |

### m2m (member-to-member)
Each member is verified against every other member in turn. The verifying member is excluded from the forecast.

### m2e (member-to-ensemble-mean)
Each member is verified against the ensemble mean (with that member excluded from the mean).

### m2c (member-to-control)
Each member is verified against the control simulation. Only available when `.add_control()` has been called.

### e2c (ensemble-mean-to-control)
The ensemble mean is verified against the control simulation. Deterministic only.

## Choosing a Comparison

### For HindcastEnsemble:
- **Deterministic metrics** → `"e2o"` (ensemble mean provides best deterministic estimate)
- **Probabilistic metrics** → `"m2o"` (need individual members for ensemble distribution)

### For PerfectModelEnsemble:
- **Standard analysis** → `"m2e"` (common default, accounts for ensemble structure)
- **Cross-validated** → `"m2m"` (more degrees of freedom, more expensive)
- **Against external reference** → `"m2c"` or `"e2c"` (requires control run)

### dim and comparison must be consistent:
```python
# CORRECT: e2o with dim="init" (no member)
hindcast.verify(metric="rmse", comparison="e2o", dim="init", ...)

# CORRECT: m2o with dim=["init", "member"]
hindcast.verify(metric="crps", comparison="m2o", dim=["init", "member"], ...)

# WRONG: e2o with dim including "member" — will error
# hindcast.verify(metric="rmse", comparison="e2o", dim=["init", "member"], ...)
```
