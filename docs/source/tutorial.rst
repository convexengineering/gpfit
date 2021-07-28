Tutorial
********

Given log-transformed data `x` and `y`, we can create a fit with `K` terms with
a single command:

.. code::

   f = ImplicitSoftmaxAffine(x, y, K)

We can then plot it:

.. code::

   f, ax = f.plot() # for 1D fits
   f, ax = f.plot_slices() # for 2D fits
   f, ax = f.plot_surface() # for 2D fits

and save it:

.. code::

   f.savetxt(filename="fit.txt") # saves string of fit to text file
   f.save(filename="fit.pkl") # saves fit object to pickle file

We can even generate a GPkit constraint set:

.. code::

   fcs = f.constraint_set()


Examples
========
Both examples come from Section 6 of this `paper
<https://dspace.mit.edu/bitstream/handle/1721.1/105753/11081_2016_9332_ReferencePDF.pdf?sequence=2&isAllowed=y>`_.

Example 1
---------

Fit convex portion of :math:`w = \frac{u^2 + 3}{(u+1)^2}` on :math:`1 \leq u \leq 3`.

.. literalinclude:: examples/ex1.py

Output:

.. literalinclude:: examples/ex1_output.txt

Example 2
---------

.. literalinclude:: examples/ex2.py

Output:

.. literalinclude:: examples/ex2_output.txt
