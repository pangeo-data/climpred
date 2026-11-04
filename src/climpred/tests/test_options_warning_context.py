"""Warning option contexts restore every setting they change."""

from itertools import product

import pytest

from climpred import set_options
from climpred.options import OPTIONS

WARNING_FLAGS = tuple(key for key in OPTIONS if key.startswith("warn_"))


@pytest.fixture(autouse=True)
def add_standard_imports():
    """These option-only tests need no downloaded ensemble fixtures."""


@pytest.fixture(autouse=True)
def restore_options():
    original = OPTIONS.copy()
    yield
    OPTIONS.clear()
    OPTIONS.update(original)


@pytest.mark.parametrize("master", [True, False])
@pytest.mark.parametrize("flags", list(product([True, False], repeat=3)))
@pytest.mark.parametrize("exception_exit", [False, True])
def test_warning_context_restores_exact_prior_state(master, flags, exception_exit):
    OPTIONS.update(dict(zip(WARNING_FLAGS, flags)))
    OPTIONS["climpred_warnings"] = master
    before = OPTIONS.copy()
    try:
        with set_options(climpred_warnings=False):
            assert all(OPTIONS[key] is False for key in WARNING_FLAGS)
            if exception_exit:
                raise RuntimeError("test context exit")
    except RuntimeError:
        if not exception_exit:
            raise
    assert OPTIONS == before, "warning context must restore every changed option"


def test_nested_warning_context_restores_each_level():
    before = OPTIONS.copy()
    with set_options(climpred_warnings=False):
        outer = OPTIONS.copy()
        with set_options(**{WARNING_FLAGS[0]: True}):
            inner = OPTIONS.copy()
            with set_options(climpred_warnings=False):
                assert not any(OPTIONS[key] for key in WARNING_FLAGS)
            assert OPTIONS == inner
        assert OPTIONS == outer
    assert OPTIONS == before, "nested warning contexts must restore each level"


def test_global_warning_switch_stays_global():
    set_options(climpred_warnings=False)
    assert OPTIONS["climpred_warnings"] is False
    assert not any(OPTIONS[key] for key in WARNING_FLAGS)


def test_unrelated_context_does_not_restore_unowned_options():
    with set_options(seasonality="season"):
        set_options(**{WARNING_FLAGS[0]: False})
    assert OPTIONS[WARNING_FLAGS[0]] is False
    assert OPTIONS["seasonality"] == "month"


def test_invalid_option_batch_does_not_change_state():
    before = OPTIONS.copy()
    with pytest.raises(ValueError):
        set_options(climpred_warnings=False, seasonality="invalid")
    assert OPTIONS == before


def test_explicit_flag_is_still_overridden_by_false_master():
    before = OPTIONS.copy()
    with set_options(climpred_warnings=False, **{WARNING_FLAGS[0]: True}):
        assert OPTIONS[WARNING_FLAGS[0]] is False
    assert OPTIONS == before
