import logging

from xarray.testing import assert_allclose, assert_equal, assert_identical


def assert_PredictionEnsemble(he, he2, how='equal', **assert_how_kwargs):
    """Loops over all datasets in PredictionEnsemble and applies assert_{how}."""

    def non_empty_datasets(he):
        """Check for same non-empty datasets."""
        return [k for k in he._datasets.keys() if he._datasets[k]]

    assert non_empty_datasets(he) == non_empty_datasets(he2)
    # check all datasets
    if how == 'equal':
        assert_func = assert_equal
    elif how == 'allclose':
        assert_func = assert_allclose
    elif how == 'identical':
        assert_func = assert_identical

    for dataset in he._datasets:
        if he._datasets[dataset]:
            if dataset == 'observations':
                for obs_dataset in he._datasets['observations']:
                    logging.info('check observations', obs_dataset)
                    assert_func(
                        he2._datasets['observations'][obs_dataset],
                        he2._datasets['observations'][obs_dataset],
                        **assert_how_kwargs,
                    )
            else:
                logging.info('check', dataset)
                assert_func(
                    he2._datasets[dataset], he._datasets[dataset], **assert_how_kwargs
                )


def check_dataset_dims_and_data_vars(before, after, dataset):
    if dataset not in ['initialized', 'uninitialized', 'control']:
        before = before._datasets['observations'][dataset]
        after = after._datasets['observations'][dataset]
    else:
        before = before._datasets[dataset]
        after = after._datasets[dataset]
    assert before.dims == after.dims
    assert list(before.data_vars) == list(after.data_vars)
