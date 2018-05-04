Motivation
==========

We wanted a Python implementation of nearest-neighbours (knn) spatial joins on large DataFrames. To make it efficient, we want to make use of spatial indexation. As of the time of writing, there is to our knowledge no such implementation available. For comparison, the popular package `GeoPandas` implements intersectio, within and contains joins.

In order to optimise a knn join with indexing, we needed access to the index object to provide a nearest search that would leave traces behind and that iterates through candidates (lazy evaluation).  
Pure Python implementations of spatial indexes allow to have such finer control over the index compared to a C library wrapper like `rtree`. It moreover transforms an external dependency on a C library to a dependency on a Python package, which one might prefer.
