.. image:: _static/logo_datastorm.png
   :align: center

Welcome to SpatialIndex's documentation!
========================================

SpatialIndex provides in-memory solutions to:
    * spatial indexing via different types of bounding volumes hierarchies.
    * true nearest-neighbors (knn) queries and joins, optimised using a spatial index. 

We'd like to warn users that the package is at a very early stage of development. We suggest not using it in production environment yet!

Introduction
------------
Given a geometry :math:`g` and a collection :math:`C` of geometries, a nearest-neighbours query asks for the :math:`k` nearest geometries in :math:`C` to :math:`g`. Different adjective are used to describe which distance is minimized:
    * Weak kNN-join: distance between the centroids of the underlying geometries.
    * Approximate kNN-join: distance between bounding geometries to the underlying geometries.
    * True kNN-join: distance between the geometries themselves.

The brute force solution computes all pairwise distances between the geometries. It is of linear complexity, which is prohibitive when looping over many reference geometries like in spatial joins. Using an acceleration data structure such as a spatial index, we have in practice log complexity (although this is not theoretically assured). Acceleration can thus lead to impressive gains on large datasets.

Nearest-neighbours search has been classically applied to points and has many implementations available (e.g. in scikit-learn). Accelerations are provided via data-structures such as kd-trees and ball-trees. 

However, spatial data can include other shapes. This package implements an in-memory spatial index assisted true kNN query/join between arbitrary geometries. It is similar to the recent PostgreSQL (9.5+) <-> operator solution.

.. toctree::
    :maxdepth: 2
    :caption: Table of Contents

    quickstart
    indexation
    api
    develop


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
