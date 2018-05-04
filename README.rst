Introduction
============

The main novelty is an in-memory solution to the true nearest-neighbours spatial joins optimized using spatial indexing.
For comparison, GeoPandas at the time of writing implements contains, within and intersects types of joins using the `rtree` indexing package.

Spatial indexes can only return candidates for nearest-neighbours in approximate order by distance. In contrast, a true nearest-neighbours query seeks the nearest-neighbours ordered by the actual shortest distance. However, care has to be taken since in general, we neither know how many neighbours will be needed to find the nearest one, nor do we have a good upper bound on its distance. 


Example
=======

The following is an example of usage of the core package::

    # Loading data
    import geopandas
    roads = geopandas.read_file("data/roads.shp")
    rails = geopandas.read_file("data/railroads.shp")

    # True-knn join. Indexe is created automatically.
    import spindex.pandas
    spindex.pandas.st_join(rails, roads, n_neighbours=2)
    

Documentation
=============

The documentation of the package will be available shortly.
