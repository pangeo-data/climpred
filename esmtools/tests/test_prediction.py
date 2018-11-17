"""Tests for predictions."""

import esmtools as et
import numpy as np
#import pandas as pd
import xarray as xr
from nose.tools import assert_equal

ds = et.prediction.load_dataset('PM_MPI-ESM-LR_ds')
ds = ds.assign(member=np.arange(10))
control = et.prediction.load_dataset('PM_MPI-ESM-LR_control')


def test_drop_members_length():
    drop = [3]
    before = list(ds.member.values)
    after = list(et.prediction.drop_members(ds, drop).member.values)
    assert_equal(len(before), len(after) + len(drop), 'not same length')


def test_drop_members_values_identical():
    drop = [3]
    before = list(ds.member.values)
    after = list(et.prediction.drop_members(ds, drop).member.values)
    assert_equal(len(before), len(after) + len(drop), 'not same length')
    before.remove(drop)
    assert_equal(before, after, 'not equal')


def test_drop_ensembles_length():
    drop = [3014]
    before = list(ds.ensemble.values)
    after = list(et.prediction.drop_ensembles(ds, drop).ensemble.values)
    assert_equal(len(before), len(after) + len(drop), 'not same length')


def test_drop_ensembles_values_identical():
    drop = [3014]
    before = list(ds.ensemble.values)
    after = list(et.prediction.drop_ensembles(ds, drop).ensemble.values)
    before.remove(drop)
    assert_equal(before, after, 'not equal')


def test_select_members_ensembles_length():
    sel_member = [1, 4, 6, 7, 8]
    before = list(ds.member.values)
    after = list(et.prediction.select_members_ensembles(
        ds, m=sel_member).member.values)
    assert_equal(len(before), len(after) + len(sel_member))


def test_select_members_ensembles_identical_members():
    sel_member = [1, 4, 6, 7, 8]
    before = list(ds.member.values)
    after = list(et.prediction.select_members_ensembles
                 (ds, m=sel_member).member.values)
    all_left = list(ds.member.values)
    for m in sel_member:
        all_left.remove(m)
    combined = (after + all_left)
    combined.sort()
    assert_equal((combined), before)
