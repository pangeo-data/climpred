import numpy as np
import pytest

from climpred.comparisons import (
    HINDCAST_COMPARISONS,
    PM_COMPARISONS,
    PROBABILISTIC_HINDCAST_COMPARISONS,
    PROBABILISTIC_PM_COMPARISONS,
)


def logical(ds):
    """Function that returns logical bool data."""
    return ds > ds.mean()


@pytest.fixture
def category_2_edges_ref():
    """Edges for contingency."""
    return np.linspace(-2, 2, 3)


@pytest.fixture
def category_2_edges_cmp():
    """Edges for contingency."""
    return np.linspace(-2, 2, 3)


@pytest.fixture
def bins():
    """Bins of actual PM North Atlantic data."""
    return np.linspace(10, 11, 11)


@pytest.fixture
def bins_hindcast():
    """Bins of actual data for offset reduced hindcast."""
    return np.linspace(-0.4, 0.6, 11)


@pytest.fixture
def probability_bins():
    """Probabilities"""
    return np.linspace(0, 1, 5)


@pytest.mark.parametrize(
    'metric',
    ['rank_histogram', 'roc', 'rps', 'Brier_score', 'discrimination', 'reliability'],
)
@pytest.mark.parametrize('comparison', PROBABILISTIC_PM_COMPARISONS)
def test_probabilistic_perfect_model(
    perfectModelEnsemble_initialized_control_1d,
    metric,
    comparison,
    probability_bins,
    bins,
):
    """Test that PerfectModelEnsemble verifies probabilistic doppyo metrics."""
    kwargs = {}
    if metric in ['rps']:
        kwargs = {'bins': bins}
    elif metric in ['roc', 'discrimination', 'reliability', 'Brier_score']:
        kwargs = {'probability_bins': probability_bins, 'logical': logical}

    perfectModelEnsemble_initialized_control_1d.verify(
        metric=metric, comparison=comparison, dim=[], **kwargs
    )


@pytest.mark.parametrize(
    'metric',
    ['rank_histogram', 'roc', 'rps', 'Brier_score', 'discrimination', 'reliability'],
)
@pytest.mark.parametrize('comparison', PROBABILISTIC_HINDCAST_COMPARISONS)
def test_probabilistic_hindcast(
    hindcast_recon_1d_ym, metric, comparison, probability_bins, bins_hindcast
):
    """Test that HindcastEnsemble verifies probabilistic doppyo metrics."""
    kwargs = {}
    if metric in ['rps']:
        kwargs = {'bins': bins_hindcast}
    elif metric in ['roc', 'discrimination', 'reliability', 'Brier_score']:
        kwargs = {'probability_bins': probability_bins, 'logical': logical}
    # reduce huge offset
    hindcast_recon_1d_ym = hindcast_recon_1d_ym - hindcast_recon_1d_ym.sel(
        time=slice('1964', '2014')
    ).mean('time').sel(init=slice('1964', '2014')).mean('init')
    hindcast_recon_1d_ym.verify(metric=metric, comparison=comparison, dim=[], **kwargs)


@pytest.mark.parametrize('metric', ['contingency', 'Heidke_score'])
@pytest.mark.parametrize('comparison', PM_COMPARISONS)
def test_contingency_metrics_perfect_model(
    perfectModelEnsemble_initialized_control_1d,
    category_2_edges_cmp,
    category_2_edges_ref,
    metric,
    comparison,
):
    """Test that PerfectModelEnsemble verifies doppyo contingency metrics."""
    perfectModelEnsemble_initialized_control_1d.verify(
        metric=metric,
        comparison=comparison,
        category_edges_cmp=category_2_edges_cmp,
        category_edges_ref=category_2_edges_ref,
    )


@pytest.mark.parametrize('metric', ['contingency', 'Heidke_score'])
@pytest.mark.parametrize('comparison', HINDCAST_COMPARISONS)
def test_contingency_metrics_hindcast(
    hindcast_recon_1d_ym,
    category_2_edges_cmp,
    category_2_edges_ref,
    metric,
    comparison,
):
    """Test that HindcastEnsemble verifies doppyo contingency metrics."""
    hindcast_recon_1d_ym.verify(
        metric=metric,
        comparison=comparison,
        category_edges_cmp=category_2_edges_cmp,
        category_edges_ref=category_2_edges_ref,
    )
