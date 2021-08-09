GP-Compatible Fitting 101
*************************

What is Geometric Programming (GP)?
===================================

For an introduction to GP, please take a look at the `GPkit documentation <http://gpkit.readthedocs.org/en/latest/gp101.html>`_.

What is GP-compatible fitting?
==============================

GP-compatible fitting is the process of fitting GP-compatible functions to data. The expression *GP-compatible* is used to describe constraints that can be written in the form of either a posynomial inequality or a monomial equality. It can also be used to describe a posynomial objective function. 

What does GPfit do?
===================

GPfit can be used to approximate a set of multivariate data points

.. math::
    
    (\mathbf{u}_i, w_i) \in R^d_{++} \times R_{++}   \hspace{1cm}     i =1...m

with a GP-compatible function, where :math:`\mathbf{u}_i` are independent variable data, :math:`w_i` are dependent variable data, and :math:`m` is the number of data points.

.. _logtransform:

More specifically, GPfit takes logarithmically transformed data

.. math::

   \mathbf{x} = \log(\mathbf{u})\\
   y = \log(w)

as an input and returns a Fit object as an output. This can be used to generate a `GPkit <http://gpkit.readthedocs.org>`_ constraint object which can be used as part of a `GPkit <http://gpkit.readthedocs.org>`_ model to solve a geometric program. This fit can also be printed, saved, and (if :math:`d \leq 2`) plotted alongside the original data.

GPfit is intended for use with data that can be well approximated by a log-convex function. A function is *log-convex* if it is convex after both the independent and dependent variables are transformed into logarithmic space.

GPfit can fit three classes of GP-compatible functions to data: Implicit Softmax Affine (ISMA) functions, Softmax Affine (SMA) functions, and Max Affine (MA) functions.

Function Classes
================

Implicit softmax affine (ISMA) functions
++++++++++++++++++++++++++++++++++++++++

An implicit softmax affine function has the form:

.. math::

   1 = \sum_{k=1}^K \frac{e^{\alpha_k b_k}}{w^{\alpha_k}} \prod_{i=1}^d u_i^{\alpha_k a_{ik}}.

GPfit calculates a function fit of the form above and then translates this to a GP-compatible posynomial inequality constraint of the form:

.. math::

   1 \geq \sum_{k=1}^K \frac{e^{\alpha_k b_k}}{w^{\alpha_k}} \prod_{i=1}^d u_i^{\alpha_k a_{ik}}

If a user specifies K = 1, GPfit will automatically return an equality constraint of the form:

.. math::

   w^{\alpha} = e^{\alpha b} \prod_{i=1}^d u_i^{\alpha a_{i}}

because this form of constraint is still GP-compatible and it is assumed that fits with one term will only be requested when an equality constraint is desired.

The ISMA function is the most general and expressive of the three function classes that GPfit uses. For a given fitting problem and number of terms, there always exists an ISMA function with at least as small a residual as the best correpsonding fit from both the softmax affine and max affine function classes. It is therefore the default choice of function for GPfit.

Softmax affine (SMA) functions
++++++++++++++++++++++++++++++

A softmax affine function has the form:

.. math::

   w^{\alpha} = \sum_{k=1}^K e^{\alpha b_k} \prod_{i=1}^d u_i^{\alpha a_{ik}}.

GPfit calculates a function fit of the form above and then translates this to a GP-compatible posynomial inequality constraint of the form:

.. math::

    w^{\alpha} \geq \sum_{k=1}^K e^{\alpha b_k} \prod_{i=1}^d u_i^{\alpha a_{ik}}.

If a user specifies K = 1, GPfit will automatically return an equality constraint of the form:

.. math::

    w^{\alpha} = e^{\alpha b} \prod_{i=1}^d u_i^{\alpha a_{i}}

because this form of constraint is still GP-compatible and it is assumed that fits with one term will only be requested when an equality constraint is desired.

For a given fitting problem and number of terms, there always exists a softmax affine function with at least as small a residual as the best possible max affine fit.


Max affine (MA) functions
+++++++++++++++++++++++++

A max affine function has the form:

.. math::

   w = \max_{k=1..K} \left[ e^{b_k} \prod_{i=1}^d u_i^{a_{ik}} \right].

GPfit calculates a function fit of the form above and then translates this to a set of GP-compatible monomial inequality constraints of the form:

.. math::

   w \geq  e^{b_k} \prod_{i=1}^d u_i^{a_{ik}}, \hspace{1cm} k = 1 ... K.

If a user specifies K = 1, GPfit will automatically return an equality constraint of the form:

.. math::

    w =  e^{b} \prod_{i=1}^d u_i^{a_{i}}

Where can I learn more?
=======================

To learn more about fitting GP-compatible models to data, take a look at the following resources:

    * `Fitting geometric programming models to data <http://web.mit.edu/~whoburg/www/papers/gpfitting.pdf>`_, by W. Hoburg, P. Kirschen, and P. Abbeel.
