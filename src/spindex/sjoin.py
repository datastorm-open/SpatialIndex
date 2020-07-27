import functools

import numpy
import pandas
import pygeos

from . import envelope, str


def sjoin(left, right, predicate,
          index_split_size=16, index_max_top_size=1,
          suffixes=("", "_right"), keep_index=False,
          ):
    join_fnc = functools.partial(_pred_join, predicate=predicate)
    return _join(left, right, join_fnc, index_split_size, index_max_top_size,
                 suffixes, keep_join_measure=False, keep_index=keep_index)


def max_join(left, right, measure,
          index_split_size=16, index_max_top_size=1,
          suffixes=("", "_right"),
          include_measure=False, keep_index=False,
          ):
    join_fnc = functools.partial(_max_intersection, measure=measure)
    return _join(left, right, join_fnc, index_split_size, index_max_top_size,
                 suffixes, include_measure, keep_index)


def knn_join(left, right, knn=1,
             index_split_size=16, index_max_top_size=1,
             suffixes=("", "_right"), include_distances=False,
             keep_index=False,
             ):
    join_fnc = functools.partial(_knn_join, knn=knn)
    return _join(left, right, join_fnc, index_split_size, index_max_top_size,
                 suffixes, keep_join_measure=include_distances,
                 keep_index=keep_index)


def _join(left, right, join_fnc,
          index_split_size, index_max_top_size,
          suffixes, keep_join_measure, keep_index,
          ):
    """
    Args:
        join_fnc (callable): The fonction giving matching candidates' indices.
            Takes 2 envelopes and a BVH index and returns the array of indices
            in a non-sparse format.
    """
    # === Prepare geometries and spatial index
    lpgeoms = envelope.AAMBRVect(left.geometry.bounds, interleaved=False)
    rpgeoms = envelope.AAMBRVect(right.geometry.bounds, interleaved=False)
    sidx = str.sort_tile_recurse(
        rpgeoms, page_size=index_split_size, max_top_size=index_max_top_size)

    # A Nx2 array of pairs of left, right indices
    # Optional third column with join_measures
    join_idx = join_fnc(left.geometry, right.geometry, lpgeoms, sidx)

    # === Format the result back in a (Geo)DataFrame

    # Keep index info and reset it to RangeIndex
    matching = ~numpy.isnan(join_idx[:, 1])
    left_res = left.iloc[join_idx[matching, 0]]
    left_idx = left_res.index.to_frame()
    left_idx.index = pandas.RangeIndex(left_res.shape[0])
    left_res.index =  pandas.RangeIndex(left_res.shape[0])
    right_res = right.iloc[join_idx[matching, 1]]
    right_idx = right_res.index.to_frame()
    right_idx.index = pandas.RangeIndex(right_res.shape[0])
    right_res.index = pandas.RangeIndex(right_res.shape[0])

    # Add suffixes on  column names present on both sides.
    common_colnames = left.columns.intersection(right.columns)
    left_res = left_res.rename(columns={col: "{}{}".format(col, suffixes[0])
                                        for col in common_colnames})
    right_res = right_res.rename(columns={col: "{}{}".format(col, suffixes[1])
                                          for col in common_colnames})
    # Concatenate result together
    frames = [left_res, right_res]
    if keep_join_measure:
        measures = pandas.DataFrame(join_idx[:, 2])
        measures.index = pandas.RangeIndex(measures.shape[0])
        frames.append(measures)
    result = pandas.concat(frames, axis=1)
    if keep_index:
        result.index = pandas.MultiIndex.from_frame(
            pandas.concat([left_idx, right_idx], axis=1))
    else:
        result.index = pandas.RangeIndex(result.shape[0])
    return result


def _pred_join(lgeoms, rgeoms, lpgeoms, sidx, predicate):
    candidates = sidx.query(lpgeoms, predicate=predicate, format="long")
    matches = getattr(lgeoms.iloc[candidates[:, 0]].values, predicate)(
        rgeoms.iloc[candidates[:, 1]].values)
    # matches = [  # un-vectorized version
    #     getattr(g, predicate)(h)
    #     for g, h in zip(lgeoms.iloc[candidates[:, 0]],
    #                     rgeoms.iloc[candidates[:, 1]])
    # ]
    return candidates[matches]


def _max_intersection(lgeoms, rgeoms, lpgeoms, sidx, measure):
    def best(lidx, ridx):
        geoms = pygeos.intersection(
            lgeoms.iloc[lidx].values.data.reshape(-1, 1),
            rgeoms.iloc[ridx].values.data.reshape(1, -1)
        )
        measures = getattr(pygeos, measure)(geoms)
        idx = measures.argmax(axis=1)
        return (
            lidx, ridx[idx], measures[numpy.arange(measures.shape[0]), idx])
        
    candidates = sidx.query(lpgeoms, predicate="intersects", format="sparse")
    result = numpy.zeros(shape=(lgeoms.shape[0], 3), dtype=float)
    cursor = 0
    for path in candidates:
        if len(path.target) > 0:
            lidx, ridx, measures = best(path.query, path.target)
        else:
            lidx, ridx, measures = path.query, numpy.nan, numpy.nan
        end = len(lidx) + cursor
        result[cursor:end, 0] = lidx
        result[cursor:end, 1] = ridx
        result[cursor:end, 2] = measures
        cursor = end
    return result


def _knn_join(lgeoms, rgeoms, lpgeoms, sidx, knn):
    def knn_best(i, geom, ridx):
        dist = numpy.array([
            (i, idx, geom.distance(ogeom))
            for (idx, ogeom) in zip(ridx, rgeoms.iloc[ridx])
        ])
        if dist.shape[0] > knn:
            dist = dist[numpy.argpartition(dist[:, 2], kth=knn)[:knn]]
        return numpy.sort(dist)

    candidates = sidx.nearest(lpgeoms, knn=knn, format="wide")
    wide_result = [knn_best(i, g, ridx)
                   for i, (g, ridx) in enumerate(zip(lgeoms, candidates))]
    return numpy.concatenate(wide_result, axis=0)
