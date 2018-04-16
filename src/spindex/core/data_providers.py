# Copyright (C) 2018 DataStorm
#
# This file is part of SpatialIndex.
#
# SpatialIndex is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SpatialIndex is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# A copy of the GNU General Public License is available in the LICENSE
# file or at <http://www.gnu.org/licenses/>.
"""
Data containers for a spatially indexed collection of geometric shapes.

The module aim to be compatible with many type of data and to evaluate lazily
as much as possible. For example, :class:`GIShapes` data attribute can be
any container implementing a subset of the Python mapping API. These can be
dictionaries, pandas Series or pandas GeoSeries for example, but also an
iterator on these types.
"""
import abc
import collections
import heapq

import toolz

import spindex.core.enclosing_geometry
import spindex.core.spatial_index


# Algorithms are common to all types of containers. They are implemented in
# the BaseShapes class inherited by the shape containers.
class BaseShapes(abc.ABC):
    """Base algorithms on indexed shapes."""
    __slots__ = ('data', 'itree', '_enclose_class')

    @abc.abstractmethod
    def __getitem__(self, idx):
        pass

    # true_nearest asks the index for candidates, computes true distances and
    # keeps n_neighbours best results so far on a fixed length max heap
    # (by distance). The index' candidates are themselves ordered by lower
    # bound on distance to the left geometry, so we can stop whenever that
    # lower_bound is higher than the max on the heap.
    def true_nearest(self, geom, n_neighbours=1):
        """Returns the true `n_neighbours`-nearest neighbours to `geom`.

        True nearest neighbours are sorted according to the actual distance
        between the geometries as given by their distance method.

        Parameters
        ----------
        geom: shapely geometry
            The target geometry to which we provide neighbours.
        n_neighbours: int
            Then number of neighbours to return.

        Returns
        -------
        list
            List of the :param:`n_neighbours` nearest neighbours to
            :param:`geom`.
        """
        egeom = self._enclose_class(geom)
        candidates = self.itree.approx_nearest(egeom)
        # Initialise the heap with n_neighbours elements
        heap = []
        for idx, _ in toolz.take(n_neighbours, candidates):
            ogeom = self[idx]
            dist = geom.distance(ogeom)
            heapq.heappush(heap, (-dist, idx))  # -dist to keep a min heap.
        # Keep only best n_neighbours elements until mindist is too high.
        for idx, mindist in candidates:
            if mindist >= -heap[0][0]:
                break
            ogeom = self[idx]
            dist = geom.distance(ogeom)
            heapq.heappushpop(heap, (-dist, idx))
        sorted_heap = [heapq.heappop(heap) for _ in range(len(heap))][::-1]
        return [(i, -d) for d, i in sorted_heap]


class GIShapes(BaseShapes):
    """
    In-memory container of spatially indexed shapes.

    Attributes
    ----------
    data: mapping to shapes
        A mapping (dict-like) object to shapes.
    itree: IndexTree
        A spatial index over `data` containing indices of the mapping.
    _enclose_class:
        A concrete enclosing geometry used in the IndexTree.
    """
    __slots__ = ()

    def __init__(self, data,
                 itree=spindex.core.spatial_index.DKMeans(),
                 ecls=spindex.core.enclosing_geometry.Rect):
        self.data = data
        self.itree = itree
        self._enclose_class = ecls

    def __repr__(self):
        return "<{} object at 0x{}>".format(self.__class__.__name__, id(self))

    def __str__(self):
        return (self.data.__str__()
                + "Spatial index: %s" % self.itree.__class__.__name__
                + "Enclosing geometry: %s" % self._enclose_class.__name__
                )

    # Data should be an iterable. Iseries iterates over its data.
    def __iter__(self):
        return iter(self.data)

    # Don't necessarily expect data to be an iterator.
    # Perfectly fine for this to raise an exception.
    def __next__(self):
        return next(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def keys(self):
        return self.data.keys()

    def values(self):
        try:
            return self.data.values()
        except TypeError:
            pass
        return self.data.values

    def items(self):
        return self.data.items()

    def get(self, idx):
        return self.data.get(idx)

    # We assume that data fits into memory.
    def create_index(self):
        pgeoms = collections.OrderedDict(
            [(k, self._enclose_class(v)) for k, v in self.data.items()]
        )
        self.itree.bulk_update(pgeoms)
