import numpy as np
import pytest
import xarray as xr
import xskillscore as xs
from xarray.testing import assert_allclose

from climpred.bootstrap import bootstrap_perfect_model
from climpred.comparisons import PM_COMPARISONS, PROBABILISTIC_PM_COMPARISONS
from climpred.metrics import __ALL_METRICS__ as all_metrics, Metric, __pearson_r
from climpred.prediction import compute_hindcast, compute_perfect_model


def logical(ds):
    return ds > ds.mean()


@pytest.fixture
def category_2_edges_ref():
    return np.linspace(-2, 2, 3)


@pytest.fixture
def category_2_edges_cmp():
    return np.linspace(-2, 2, 3)


@pytest.fixture
def bins():
    return np.linspace(-9, 11, 10)


@pytest.fixture
def probability_bin_edges():
    return np.linspace(0, 1, 5)


@pytest.mark.parametrize('metric', ['rank_histogram', 'roc'])
@pytest.mark.parametrize('comparison', PROBABILISTIC_PM_COMPARISONS)
def test_probabilistic(perfectModelEnsemble_initialized_control_1d, metric, comparison):
    """Test ."""
    perfectModelEnsemble_initialized_control_1d.verify(
        metric=metric, comparison=comparison, dim=[]
    )


@pytest.mark.parametrize('metric', ['contingency', 'Heidke_score'])
@pytest.mark.parametrize('comparison', PM_COMPARISONS)
def test_contingency_metrics(
    perfectModelEnsemble_initialized_control_1d,
    category_2_edges_cmp,
    category_2_edges_ref,
    metric,
    comparison,
):
    """Test ."""
    perfectModelEnsemble_initialized_control_1d.verify(
        metric=metric,
        comparison=comparison,
        category_edges_cmp=category_2_edges_cmp,
        category_edges_ref=category_2_edges_ref,
    )
