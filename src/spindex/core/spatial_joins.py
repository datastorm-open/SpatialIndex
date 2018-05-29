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
'''
Spatial joins using indexing.
'''
import pdb
import heapq
import multiprocessing

import numpy
import toolz

import spindex.core.enclosing_geometry
import spindex.core.spatial_index


HOW_ARGLIST = ['left']
OP_ARGLIST = ['knn', 'max_intersection']


def nearest(geom, right, right_sindex, n_neighbours=1):
    visitor = spindex.core.spatial_index.ApproxNearest()
    bgeom = spindex.core.enclosing_geometry.Rect(geom.bounds)
    candidates = right_sindex.accept(visitor, bgeom)

    # Initialise the heap with n_neighbours elements
    heap = list(map(lambda x: (-geom.distance(right[x[0]]), x[0]),
                    [next(candidates) for _ in range(n_neighbours)]))

    # Keep only best n_neighbours elements until mindist is too high.
    for idx, mindist in candidates:
        if mindist >= -heap[0][0]:
            break
        dist = geom.distance(right[idx])
        heapq.heappushpop(heap, (-dist, idx))

    # Return the sorted best neighbours.
    sorted_heap = [heapq.heappop(heap) for _ in range(len(heap))][::-1]
    return [(i, -d) for d, i in sorted_heap]


def nearest_join(left, right, n_neighbours=1, n_jobs=1, chunk_size=1000):
    bgeoms = {idx: spindex.core.enclosing_geometry.Rect(geom.bounds)
              for idx, geom in right.items()}
    right_sindex = spindex.core.spatial_index.BoundingGeometryTree()
    right_sindex.accept(
        spindex.core.spatial_index.DKMeansBulkInsert(n_jobs=n_jobs), bgeoms)
    if n_jobs == 1:
        res = _nearest_task(left, right, right_sindex,
                            n_neighbours=n_neighbours)
    else:
        chunks = [toolz.partition_all(chunk_size, left.items())]
        task = toolz.partial(_nearest_task, right=right,
                             n_neighbours=n_neighbours,
                             right_sindex=right_sindex)
        with multiprocessing.Pool(n_jobs) as pool:
            res = pool.map(task, chunks)
    return numpy.array(res).reshape(-1, 3)


def _nearest_task(left, right, right_sindex, n_neighbours=1):
    return [(idx, jdx, dist) for idx, geom in left.items()
            for jdx, dist in nearest(geom, right, right_sindex,
                                     n_neighbours=n_neighbours)]


# Interface function: interprete arguments and delegate appropriately
def st_join(left, right, how='left', op='knn', n_jobs=1, **kwargs):
    '''
    Spatial joins optimised with indexing on the right series.

    A spatial join is a join based on a spatially defined predicate.
    The following types of joins are supported:

        1. Nearest neighbours (knn): match

    A spatial index on `right` is used to considerably limit the candidate
    matches in the right series. The types of joins supported are nearest
    neighbours (knn).

    Parameters
    ----------
    left: mapping to Shapely geometries
    right: spindex GIShapes
    how: str (default 'left')
        'left' or 'inner' for the type of join. Left joins are implemented so
        far.
    op: str (default 'knn')
        * 'knn': true nearest-neighbours join. For each left geometry, it
           finds the actual `n_neighbours`-nearest neighbours in the metric
           given by the distance between shapes.
    kwargs:
        keyword arguments depending on `op`:
            * 'knn': n_neighbours, int (default 1).

    Yields
    ------
    List
        The non-parallel version, generates for each left geometry a list of
        length `n_neighbours` of 2-tuples of index and distance of/to the
        nearest-neighbours.
        The parallel version, generates a list of results as in the
        non-parallel version of length `chunk_size`.
    '''

    # how argument
    if not how in HOW_ARGLIST:
        raise ValueError("how must be one of {}".format(HOW_ARGLIST))

    # op argument
    if op == 'knn':
        n_neighbours = kwargs.get('n_neighbours', 1)
        task = toolz.partial(_st_knn, right=right,
                             how=how, n_neighbours=n_neighbours)
    elif op == 'max_intersection':
        raise NotImplementedError
    else:
        raise ValueError("op argument must be one of {}."
                         .format(','.join(OP_ARGLIST)))

    # n_jobs argument
    if n_jobs == 0:
        raise ValueError("n_jobs cannot be set to zero.")

    elif n_jobs < 0:
        raise NotImplementedError("Negative n_jobs are not implemented yet.")

    elif n_jobs > 1:
        task_size = kwargs.get('task_size', 10**4)
        chunk_size = kwargs.get('chunk_size', 10**5)
        left_chunks = toolz.partition_all(chunk_size, left)

        with multiprocessing.Pool(n_jobs) as pool:
            for chunk in left_chunks:
                sub_lefts = list(toolz.partition_all(task_size, chunk))
                yield list(toolz.concat(
                    pool.map(toolz.compose(list, task), sub_lefts)
                ))
                # TODO: optimize, since it re-pickles function each time

    else:  # as if n_jobs == 1
        chunk_size = kwargs.get('chunk_size', 0)
        if chunk_size > 0:
            yield from list(toolz.partition_all(chunk_size, task(left)))
        else:   # iterates
            yield from task(left)


def _st_knn(left, right, how='left', n_neighbours=1):
    '''
    True nearest neighbours join optimised using indexing.
    '''
    for k, geom in left.items():
        yield right.true_nearest(geom, n_neighbours)


def _st_max_intersection(left, right, how='left', n_jobs=1):
    raise NotImplementedError
