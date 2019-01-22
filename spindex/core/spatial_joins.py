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
import math
import multiprocessing

import numpy
import toolz

import spindex.core.enclosing_geometry
import spindex.core.spatial_index


HOW_ARGLIST = ['left']
OP_ARGLIST = ['knn', 'max_intersection']


def prepare_geoms(geoms, n_jobs=1, chunk_size=10000):
    '''TODO'''
    bgeoms = {idx: spindex.core.enclosing_geometry.Rect(geom.bounds)
              for idx, geom in geoms.items()}
    sindex = spindex.core.spatial_index.BoundingGeometryTree()
    sindex.accept(
        spindex.core.spatial_index.DKMeansBulkInsert(n_jobs=n_jobs), bgeoms)
    return (bgeoms, sindex)


def nearest(geom, right, right_sindex, n_neighbours=1):
    visitor = spindex.core.spatial_index.ApproxNearest()
    bgeom = spindex.core.enclosing_geometry.Rect(geom.bounds)
    candidates = right_sindex.accept(visitor, bgeom)

    # Initialise the heap with n_neighbours elements
    heap = []
    for _ in range(n_neighbours):
        idx, mindist = next(candidates)
        dist = geom.distance(right[idx])
        heapq.heappush(heap, (-dist, idx))
    # Keep only best n_neighbours elements until mindist is too high.
    for idx, mindist in candidates:
        if mindist >= -heap[0][0]:
            break
        dist = geom.distance(right[idx])
        heapq.heappushpop(heap, (-dist, idx))

    # Return the sorted best neighbours.
    sorted_heap = [heapq.heappop(heap) for _ in range(len(heap))][::-1]
    return [(i, -d) for d, i in sorted_heap]


def nearest_join(left, right, n_neighbours=1, sindex=None,
                 n_jobs=1, chunk_size=10000):
    if sindex is None:
        _, sindex = prepare_geoms(right, n_jobs)
    if n_jobs == 1:
        res = _nearest_task(left.items(), right, sindex,
                            n_neighbours=n_neighbours)
    else:
        chunks = list(toolz.partition_all(chunk_size, left.items()))
        task = toolz.partial(_nearest_task, right=right,
                             n_neighbours=n_neighbours,
                             right_sindex=sindex)
        with multiprocessing.Pool(n_jobs) as pool:
            res = pool.map(task, chunks)
            res = [y for x in res for y in x]
    return res
    # return numpy.array(res).reshape(-1, 3)


def _nearest_task(left, right, right_sindex, n_neighbours=1):
    return [(idx, jdx, dist) for idx, geom in left
            for jdx, dist in nearest(geom, right, right_sindex,
                                     n_neighbours=n_neighbours)]


def max_intersection(geom, right, right_sindex):
    visitor = spindex.core.spatial_index.Intersection(op='intersects')
    bgeom = spindex.core.enclosing_geometry.Rect(geom.bounds)
    candidates = right_sindex.accept(visitor, bgeom)
    if geom.type in ("LineString", "MultiLineString"):
        measures = [(geom.intersection(right[x]).length, x)
                    for x in candidates]
    elif geom.type in ("Polygon", "MultiPolygon"):
        measures = [(geom.intersection(right[x]).area, x)
                    for x in candidates]
    else:
        raise ValueError("geometry type %s is not supported" % geom.type)
    if len(measures) == 0:
        return [(None, numpy.nan)]
    length, _ = max(measures)
    if math.isclose(length, 0.):
        return [(None, numpy.nan)]
    return [(x, l) for l, x in measures if math.isclose(l, length)]


def max_intersection_join(left, right, sindex=None,
                          n_jobs=1, chunk_size=10000):
    if sindex is None:
        _, sindex = prepare_geoms(right, n_jobs)
    if n_jobs == 1:
        res = _max_intersection_task(left.items(), right, sindex)
    else:
        chunks = list(toolz.partition_all(chunk_size, left.items()))
        task = toolz.partial(_max_intersection_task, right=right,
                             right_sindex=sindex)
        with multiprocessing.Pool(n_jobs) as pool:
            res = pool.map(task, chunks)
            res = [y for x in res for y in x]
    return res
#    return numpy.array(res).reshape(-1, 3)


def _max_intersection_task(left, right, right_sindex):
    return [(idx, jdx, measure) for idx, geom in left
            for jdx, measure in max_intersection(geom, right, right_sindex)]
