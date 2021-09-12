Developpement
=============

.. rubric:: Design Goals

We have set a few goals for the API:

    #. Use the most common interfaces possible. For example, GIShapes take for data input anything that implements a subset of the mapping API.
    #. Make use of generators for lazy evaluation whenever it makes sense. For example, an index approximate nearest query generates candidates one after the other, which is perfect for true-knn queries since we do not know how many candidates will be required.
