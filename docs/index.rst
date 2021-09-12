.. image:: _static/logo_datastorm.png
   :align: center

Welcome to SpatialIndex's documentation!
========================================

SpatialIndex provides in-memory solutions to:
    * spatial indexing
    * true nearest-neighbours (knn) queries and joins, optimised using a spatial index.

A true knn query is one that asks for the geometries minimising the actual distance to the reference geometry, as opposed to minimising an approximate distance like spatial indexes do. Computing the distance to all possible geometries is of linear complexity, which is prohibitive when looping over many reference geometries like in joins. Using a spatial index, we have in practice log complexity, although this is not theoretically assured. 

.. toctree::
    :maxdepth: 2
    :caption: Table of Contents

    motivation
    quickstart
    indexation
    api
    develop


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
