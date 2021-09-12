Introduction
============

Fast vectorised spatial indexing, done in python with numpy.

Having objects on the python side gives us more control on them from python.
For example, we implemented natively true nearest neighbours search algorithms,
true as opposed to approximate search.

For comparison, GeoPandas at the time of writing implementied a few joins
operations through the external `rtree` C library.


Example
=======

The following is an example of usage of the core package::

    # Loading data from tests directory
    import geopandas
    roads = geopandas.read_file("data/roads.shp")
    rails = geopandas.read_file("data/railroads.shp")

    # True-knn join. Indexe is created automatically.
    import spindex.joins
    spindex.joins.sjoin(rails, roads, op="knn", n_neighbours=2)
    

Documentation
=============

The documentation of the package will be available shortly.
