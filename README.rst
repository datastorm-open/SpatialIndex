Introduction
============

The main novelty of this package is an in-memory solution to the true nearest-neighbours spatial joins.
For comparison, GeoPandas at the time of writing implements contains, within and intersects types of joins.

The interface is intended to be compatible with data-structures implementing a subset of the python mappings API with values shapely geometries. These include among others dictionaries, pandas Series and geopandas GeoSeries. 

Also, the core subpackage provides lazy evaluation where possible, allowing for low-memory usage.

Example
=======

The following is an example of usage of the core package.

..ipython::

    In [1]: import geopandas
    
    In [2]: roads = geopandas.read_file("data/roads.shp")
    
    In [3]: rails = geopandas.read_file("data/railroads.shp")
    
    In [4]: import spindex
    
    In [5]: iroads = spindex.data_providers.GIShapes(roads.geometry)
    
    In [6]: irails = spindex.data_providers.GIShapes(rails.geometry)
    
    In [7]: iroads.create_index()

    In [8]: iroads.true_nearest(rails.geometry.iloc[0], n_neighbours=
	...: 3)
    Out[8]: [(35926, 0.0), (35924, 80.15879919875059), (15394, 80.15879919875059)]

The elements of the result list correspond to a neighbour and gives its index and distance to the geometry. One can also get all true neighbours for a collection via a spatial join:

..ipython::

    In [9]: gen = spindex.spatial_joins.st_join(irails, iroads, n_neighbours=2)

    In [10]: next(gen)
    Out[10]: [(35926, 0.0), (35924, 80.15879919875059)]

Evaluation is lazy and a generator is returned.

Wrappers around pandas and geopandas DataFrames are coming shortly. Spatial joins for DataFrames will return the joined DataFrame.

Documentation
=============

The documentation of the package will be available shortly.
