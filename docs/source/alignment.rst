**********************
Verification Alignment
**********************

A forecast is verified by comparing a set of initializations at a given lead to
observations over some window of time. However, there are a few ways to decide *which*
initializations or verification window to use.

One can pass the keyword ``alignment=...`` to any hindcast compute functions (e.g.,
:py:func:`~climpred.prediction.compute_persistence`,
:py:func:`~climpred.prediction.compute_hindcast`,
:py:meth:`~climpred.classes.HindcastEnsemble.verify`) to change the behavior for
aligning forecasts with the verification product. **Note that the alignment decision
only matters for `hindcast experiments <terminology.html#simulation-design>`_.
Perfect model experiments are perfectly time-aligned by design.**

The available keywords are:

*  ``'same_inits'`` (current default): Use a common set of initializations that verify
   across all leads. This ensures that there is no bias in the result due to the state
   of the system for the given initializations.


*  ``'same_verifs'``: Use a common verification window across all leads. This ensures
   that there is no bias in the result due to the observational period being verified
   against.


*  ``'maximize'``: Use all available initializations at each lead that verify against
   the observations provided. This changes both the set of initializations and the
   verification window used at each lead.

See `logging <#logging>`_ for details on how to print logs out to the
screen or to save them out as a file. These logs show which initializations and
verification dates are being used in computations.

Same Initializations
####################

(**current default**)

.. code::

    ``alignment='same_inits'``

Below is an example of the logic used in ``climpred`` to select initializations that
verify with the given verification data over all leads.

Here we have an initialized forecasting system with annual initializations from 1990
through 2000 and three lead years. We are verifying it against a product that spans 1995
through 2002.

Two conditions must be met to retain the initializations for verification:

1. All forecasted times (i.e., initialization + lead year) for a given initialization
   must be contained in the verification data. Schematically, this means that there must
   be a union between a column in the top panel and the time series in the bottom panel.
   The 2000 initialization below is left out since the verification data does not
   contain 2003.

2. There must be an observation in the verification data for the given initialization.
   In combination with (1), initializations 1990 through 1994 are left out. This logic
   exists so that any `reference forecasts <reference_forecast.html>`__
   (e.g. a persistence forecast) use an identical set of initializations as the
   initialized forecast.

.. image:: images/alignment_plots/same_inits_alignment.png

Logging
#######

``climpred`` uses the standard library ``logging`` to store the initializations and
verification dates used at each lead for a given computation. This is used internally
for testing, but more importantly, can be activated by the user so they can be sure of
how computations are being done.

To see the log interactively (e.g. while working in Jupyter notebooks or on the command
line):

.. code:: python

    import logging, sys
    from climpred.tutorial import load_dataset
    from climpred.prediction import compute_hindcast

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    hind = load_dataset('CESM-DP-SST').isel(lead=slice(0, 3))
    verif = load_dataset('FOSI-SST')
    result = compute_hindcast(hind, verif)

This prints out the following for the user:

.. code:: bash

    INFO:root:`compute_hindcast` for metric pearson_r and comparison e2o at 2020-02-29 18:39:08.439830
    ++++++++++++++++++++++++++++++++++++++++++++++
    INFO:root:lead=01 | dim=time | inits=1954-01-01 00:00:00-2014-01-01 00:00:00 | verif=1955-01-01 00:00:00-2015-01-01 00:00:00
    INFO:root:lead=02 | dim=time | inits=1954-01-01 00:00:00-2014-01-01 00:00:00 | verif=1956-01-01 00:00:00-2016-01-01 00:00:00
    INFO:root:lead=03 | dim=time | inits=1954-01-01 00:00:00-2014-01-01 00:00:00 | verif=1957-01-01 00:00:00-2017-01-01 00:00:00

To store the log as a file in which all computations will be appended to it, use the
following:

.. code:: python

    import logging
    from climpred.tutorial import load_dataset
    from climpred.prediction import compute_hindcast

    # You can name the log file anything with or without an extension.
    logging.basicConfig(filename='hindcast.log.out', level=logging.INFO)
    hind = load_dataset('CESM-DP-SST').isel(lead=slice(0, 3))
    verif = load_dataset('FOSI-SST')
    result1 = compute_hindcast(hind, verif, metric='pearson_r')
    result2 = compute_hindcast(hind, verif, metric='nmse')

This stores a file in the local directory called ``hindcast.log.out`` with the following
contents:

.. code:: bash

    $ cat hindcast.log.out
    INFO:root:`compute_hindcast` for metric pearson_r and comparison e2o at 2020-02-29 18:50:16.181650
    ++++++++++++++++++++++++++++++++++++++++++++++++
    INFO:root:lead=01 | dim=time | inits=1954-01-01 00:00:00-2014-01-01 00:00:00 | verif=1955-01-01 00:00:00-2015-01-01 00:00:00
    INFO:root:lead=02 | dim=time | inits=1954-01-01 00:00:00-2014-01-01 00:00:00 | verif=1956-01-01 00:00:00-2016-01-01 00:00:00
    INFO:root:lead=03 | dim=time | inits=1954-01-01 00:00:00-2014-01-01 00:00:00 | verif=1957-01-01 00:00:00-2017-01-01 00:00:00
    INFO:root:`compute_hindcast` for metric nmse and comparison e2o at 2020-02-29 18:50:23.844099
    ++++++++++++++++++++++++++++++++++++++++++++++++
    INFO:root:lead=01 | dim=time | inits=1954-01-01 00:00:00-2014-01-01 00:00:00 | verif=1955-01-01 00:00:00-2015-01-01 00:00:00
    INFO:root:lead=02 | dim=time | inits=1954-01-01 00:00:00-2014-01-01 00:00:00 | verif=1956-01-01 00:00:00-2016-01-01 00:00:00
    INFO:root:lead=03 | dim=time | inits=1954-01-01 00:00:00-2014-01-01 00:00:00 | verif=1957-01-01 00:00:00-2017-01-01 00:00:00
