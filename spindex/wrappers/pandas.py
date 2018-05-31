"""
Module wrapping pandas DataFrames.
"""
import numpy
import pandas

import spindex.core.data_providers
import spindex.core.spatial_joins


def st_join(lframe, rframe, op='knn', **kwargs):
    lidx = 'left_index' if lframe.index.name is None else lframe.index.name
    ridx = 'right_index' if rframe.index.name is None else rframe.index.name
    join_fnc = {
        'knn': spindex.core.spatial_joins.nearest_join,
        'max_intersection': spindex.core.spatial_joins.max_intersection_join,
    }
    if op not in join_fnc.keys():
        raise ValueError(
            "Unrecognized `op`: must be one of %s." % join_fnc.keys())
    join = join_fnc[op](lframe['geometry'], rframe['geometry'], **kwargs)
    result = (pandas.DataFrame(join, columns=[lidx, ridx, 'join_measure'])
              .merge(lframe, how='left', left_on=lidx, right_index=True,
                     suffixes=('', '_l'))
              .merge(rframe.drop('geometry', axis=1), how='left',
                     left_on=ridx, right_index=True, suffixes=('', '_r'))
              )
    return result
