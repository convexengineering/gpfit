GP-Compatible Fitting 101
*************************

What is Geometric Programming (GP)?
===================================

For an introduction to GP, please take a look at the `GPkit documentation <http://gpkit.readthedocs.org/en/latest/gp101.html>`_.

What is GP-compatible fitting?
==============================

Implicit softmax affine (ISMA) functions
++++++++++++++++++++++++++++++++++++++++

.. math::

   1 = \sum_{k=1}^K \frac{e^{\alpha_k b_k}}{w^{\alpha_k}} \prod_{i=1}^d u_i^{\alpha_k a_{ik}}

Softmax affine (SMA) functions
++++++++++++++++++++++++++++++

.. math::

   w = \sum_{k=1}^K e^{\alpha b_k} \prod_{i=1}^d u_i^{\alpha a_{ik}}

Max affine (MA) functions
+++++++++++++++++++++++++

.. math::

   w = \max_{k=1..K} \left[ e^{b_k} \prod_{i=1}^d u_i^{a_{ik}} \right]


Where can I learn more?
=======================

To learn more about fitting GP-compatible models to data, please take a look at the following resources:

    * `Fitting geometric programming models to data <http://web.mit.edu/~whoburg/www/papers/gp_fitting.pdf>`_, by W. Hoburg, P. Kirschen, and P. Abbeel.