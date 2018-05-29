Getting Started
===============

True nearest-neighbours queries seek to find in a possibly heterogeneous collection of geometries the nearest neighbours to a reference geometry in the actual distance metric. We make use of spatial indexes to optimise such queries.

Pandas Wrapper
--------------

A thin wrapper around pandas DataFrames is part of the package. DataFrames are assumed to have a column named 'geometry' consisting of Shapely's geometries. This is compatible with GeoPandas, which is used in the examples below to load datasets.


A compatible dataframe together with a spatial index is stored in a `GeoFrame` instance. It can be built as such::

    # load dataset
    import geopandas
    roads = geopandas.read_file("data/roads.shp")

    # build spatial index
    import spindex.pandas
    geo_roads = spindex.pandas.GeoFrame(roads)


GeoFrame objects are then passed to queries to make use of the spatial index, allowing to reuse the index without rebuilding it.

Spatial Joins
-------------

Joins are computed via the function st_join. The `op` argument controls the operation used to perform the join:

    * 'knn': true nearest neighbours.


.. rubric:: True Nearest Neighbours

A join query with op argument set to 'knn' will perform a true nearest-neighbours (left or inner) query.
The query result is given in pseudo-code

.. code-block:: none

    for each left geometry:
	for each neighbour in the nearest-neighbours:
	    add neighbour's attributes to left and yield


The number of neighbours is controlled by the `n_neighbours` argument. The distance between two geometries is given by Shapely's distance function: :code:`left.distance(right)`. This computes the minimum distance between the two underlying shapes `left` and `right`.

Continuing with the example above::

    rails = geopandas.read_file("data/railroads.shp")
    spindex.pandas.st_join(rails, geo_roads, op='knn', n_neighbours=2)
