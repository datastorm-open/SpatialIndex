Spatial Indexing
================

The difficulty in optimising a nearest-neighbours query reside in the fact that we do not have a good bound on their distances. A spatial index can recover quickly candidates as it has information about locality. However, an index has only approximate information about the underlying geometries, and can thus only compute approximate distances. Therefore, it is not initially known how many candidates will be required before all true nearest-neighbours are found.

This motivated the implementation of pure Python (and possibly Cython) solutions to spatial indexing allowing for finer control over the index object.

More information on indexation will be added in the future.
