# Plan to Reduce pytest Execution Time

Current state: `pytest` reduced from ~20 minutes to significantly less (estimated ~8-10 minutes) via fixture optimization.

## 1. Profiling and Measurement (Completed)
- **Slowest Tests Identified:** Bootstrap and Bias Removal tests are the primary bottlenecks.
- **Root Cause Found:** Extreme setup overhead (~0.5s per test) due to function-scoped data loading fixtures and combinatorial parameterization.

## 2. Fixture Optimization (Implemented)
- **Session Scoping:** Moved `load_dataset` fixtures in `src/climpred/conftest.py` to `session` scope. This avoids repeated NetCDF I/O and coordinate decoding.
- **Safety Balance:** Reverted `Ensemble` class fixtures (HindcastEnsemble, PerfectModelEnsemble) to `function` scope to prevent test-to-test mutation leaks, while keeping their initialization fast by using the session-cached datasets.
- **Stability:** Solved NetCDF4 segmentation faults caused by repeated file open/close cycles.

## 3. Pruning Parameterizations (Action Required)
- **`test_bootstrap.py`:** Currently tests all combinations of (metrics x alignments). Recommend testing only one metric (e.g., `crps`) across all alignments and only one alignment for other metrics.
- **`test_bias_removal.py`:** combinatorial explosion of (methods x seasonalities x alignments x cv). Recommend reducing the number of seasonalities tested for every method.
- **`test_HindcastEnsemble_class.py`:** `test_fractional_leads_lower_than_month_lead_units` runs 16 tests for coordinate logic. Can be reduced to a representative subset (e.g., 2 calendars, 2 frequencies).

## 4. Synthetic Data for Logic Tests (Action Required)
- **Identify Tests:** Many tests in `test_PredictionEnsemble.py` and `test_utils.py` use `load_dataset` but only verify metadata or simple arithmetic.
- **Refactor:** Replace these with small $2\times2$ or $3\times3$ synthetic `xarray` objects created in-memory. This eliminates the remaining overhead of the `session` fixtures for these specific tests.

## 5. Parallelization
- **pytest-xdist:** Compatibility has been improved by moving to `session` scoped data fixtures. Can now be run with `pytest -n auto` more reliably.
