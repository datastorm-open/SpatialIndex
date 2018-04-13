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
import toolz
import warnings
import multiprocessing
import heapq
import spindex.core.enclosing_geometry as prepared


HOW_ARGLIST = ['left']
OP_ARGLIST = ['knn', 'max_intersection']


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
        raise ValueError("how argument must be one of {}."
                         .format(','.join(OP_ARGLIST)))

    # n_jobs argument
    if n_jobs == 0:
        raise ValueError("Number of jobs cannot be set to zero.")

    elif n_jobs < 0:
        raise NotImplementedError("Negative number of jobs are not "
                                  "implemented yet.")

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
