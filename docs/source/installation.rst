.. _installation:

Installation
************

GPfit can be installed using pip:

.. code::

   pip install gpfit

GPfit requires the following packages:

* `GPkit <https://gpkit.readthedocs.io>`_
* numpy
* scipy
* matplotlib

GPkit requires the installation of solvers (CVXOPT and/or MOSEK).
CVXOPT is open source  and should be installed by default with GPkit.
For MOSEK installation instructions or troubleshooting help for either solver,
take a look at the `GPkit installation docs
<https://gpkit.readthedocs.io/en/latest/installation.html>`_.

To test your installation of GPfit, use:

.. code::

   pytest --pyargs gpfit

If you encounter any bugs please email ``gpkit@mit.edu``
or `raise a GitHub issue
<http://github.com/convexengineering/gpfit/issues/new>`_.

