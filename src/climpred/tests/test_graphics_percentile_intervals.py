"""Percentile confidence intervals need not enclose the original skill estimate."""

import numpy as np
import pytest
import xarray as xr

from climpred.graphics import plot_bootstrapped_skill_over_leadyear


@pytest.mark.parametrize("dataset", [False, True])
@pytest.mark.parametrize("outside", ["initialized", "persistence", "neither"])
def test_plot_keeps_interval_endpoints_and_point_estimates(dataset, outside):
    plt = pytest.importorskip("matplotlib.pyplot")
    values = np.array(
        [
            [[0.2, 0.8], [0, 0], [0.1, 0.7], [0.3, 0.9]],
            [[0.3, 0.7], [0.1, 0.2], [0.2, 0.6], [0.4, 0.8]],
        ]
    )
    index = {"initialized": 0, "persistence": 1}.get(outside)
    if index is not None:
        # First point is below its interval, second is above its interval.
        values[index, 2] = [0.4, 0.1]
        values[index, 3] = [0.6, 0.3]
    data = xr.DataArray(
        values,
        dims=("skill", "results", "lead"),
        coords={
            "skill": ["initialized", "persistence"],
            "results": ["verify skill", "p", "low_ci", "high_ci"],
            "lead": [1, 2],
        },
        name="skill_value",
        attrs={"confidence_interval_levels": "0.975-0.025", "metric": "pearson_r"},
    )
    data.lead.attrs["units"] = "years"
    obj = data.to_dataset() if dataset else data
    if dataset:
        obj.attrs = data.attrs.copy()
    before = obj.copy(deep=True)
    fig, ax = plt.subplots()
    try:
        actual = plot_bootstrapped_skill_over_leadyear(obj, ax=ax)
        assert actual is ax
        # Errorbar collections must retain the exact supplied interval endpoints.
        assert len(ax.collections) == 2
        for i, collection in enumerate(ax.collections):
            segments = collection.get_segments()
            np.testing.assert_allclose([s[:, 1] for s in segments], values[i, 2:4].T)
        # Both original point-estimate series must remain visible and labeled.
        handles, labels = ax.get_legend_handles_labels()
        assert [text.get_text() for text in ax.get_legend().get_texts()] == [
            "initialized",
            "persistence",
        ]
        handles = dict(zip(labels, handles))
        for i, label in enumerate(["initialized", "persistence"]):
            handle = handles[label]
            line = handle.lines[0] if hasattr(handle, "lines") else handle
            np.testing.assert_allclose(
                np.asarray(line.get_ydata(), dtype=float), values[i, 0]
            )
        xr.testing.assert_identical(obj, before)
    finally:
        plt.close(fig)


def test_inverted_interval_is_not_silently_corrected():
    plt = pytest.importorskip("matplotlib.pyplot")
    data = xr.DataArray(
        [[[0.2], [0], [0.6], [0.4]]],
        dims=("skill", "results", "lead"),
        coords={
            "skill": ["initialized"],
            "results": ["verify skill", "p", "low_ci", "high_ci"],
            "lead": [1],
        },
        attrs={"confidence_interval_levels": "0.975-0.025", "metric": "pearson_r"},
    )
    data.lead.attrs["units"] = "years"
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="negative"):
            plot_bootstrapped_skill_over_leadyear(data, ax=ax)
    finally:
        plt.close(fig)
