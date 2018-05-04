Developpement
=============

The package is at an early stage of development.
True kNN joins might be merged into an existing project.

.. rubric:: Goals

Provide Python/Cython solutions to the many bounding volumes hierarchies acceleration data-structures together with accelerated algorithms. 
At the moment, we focus on spatial indexing and spatial joins.

We would like to separate the following structures in order to allow for the implementation of many variants:
    * Bounding Geometries: includes AAMBR, Enclosing Spheres, Convex Hulls, ...
    * BVH-trees: data model for bounding volumes hierarchies tree-structures with abstract methods for insert, delete and their bulk versions.
    * Metrics: functions to compute distances between geometries.

We have set a few goals for the API:

    #. Use the most common interfaces possible. For example, GIShapes take for data input anything that implements a subset of the mapping API.
    #. Make use of generators for lazy evaluation whenever it makes sense. For example, an index approximate nearest query generates candidates one after the other, which is perfect for true-knn queries since we do not know how many candidates will be required.
