"""
Module wrapping pandas DataFrames.
"""
import numpy
import pandas

import spindex.core.spatial_joins


def st_join(lframe, rframe, op='knn', how='left',
            join_info=False, sindex=None, **kwargs):
    lidx = 'left_index' if lframe.index.name is None else lframe.index.name
    ridx = 'right_index' if rframe.index.name is None else rframe.index.name
    join_fnc = {
        'knn': spindex.core.spatial_joins.nearest_join,
        'max_intersection': spindex.core.spatial_joins.max_intersection_join,
    }
    if op not in join_fnc.keys():
        raise ValueError(
            "Unrecognized `op`: must be one of %s." % join_fnc.keys())
    join = join_fnc[op](lframe['geometry'], rframe['geometry'],
                        sindex=sindex, **kwargs)
    if how == 'left':
        rframe = rframe.drop('geometry', axis=1)
    elif how == 'right':
        lframe = lframe.drop('geometry', axis=1)
    elif how in ('inner', 'outer'):
        raise NotImplementedError
    else:
        raise ValueError("how must be one of {'left', 'right', 'inner', "
                         "'outer'}")
    result = (pandas.DataFrame(join, columns=[lidx, ridx, 'join_measure'])
              .merge(lframe, how='left', left_on=lidx, right_index=True,
                     suffixes=('', '_l'))
              .merge(rframe, how='left', left_on=ridx, right_index=True,
                     suffixes=('', '_r'))
              )
    if not join_info:
        result.drop([lidx, ridx, 'join_measure'], axis=1, inplace=True)
    return result
