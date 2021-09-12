"""
Spatial Joins for GeoPandas GeometricArrays
"""
import numpy
import pandas

from . import envelope
from . import build


def prepare(geometry):
    if isinstance(geometry, envelope.AAMBRs):
        return geometry
    bds = geometry.bounds
    return envelope.AAMBRs(bds[:, :2], bds[:, 2:])


def spjoin(left, right, op, right_sidx=None, **kwargs):
    # Dealing with arguments
    valid_ops = ("knn", "max_intersection")
    if op not in valid_ops:
        raise ValueError("op {} is not supported, should be one of {}"
                         .format(op, valid_ops))
    if op == "knn":
        knn = kwargs.get("n_neighbours", 1)

    if right_sidx is None:
        right_sidx = build.sort_tile_recurse(prepare(right.geometry.values))

    # True matchings
    pgeoms = prepare(left.geometry.values)
    candidates = right_sidx.nearest(pgeoms, knn=knn)
    index, measures = [], []
    for idx, g in zip(candidates, left.geometry.values):
        if op == "knn":
            measure = right.geometry.values[idx].distance(g)
            jdx = numpy.argpartition(measure, knn-1)[:knn]
        else:
            measure = right.geometry.values[idx].intersection(g).length
            jdx = [numpy.argmax(measure)]
        index.append(idx[jdx])
        measures.append(measure[jdx])
    match_index = numpy.array(index)
    match_measure = numpy.array(measures)

    # Utilisation d'un RangeIndex par défaut
    left = left.copy()
    right = right.copy()
    if left.index.names == [None]:
        left.index.name = "index_left"
    if right.index.names == [None]:
        right.index.name = "index_right"
    lindex_names = left.index.names
    left = left.reset_index()
    rindex_names = right.index.names
    right = right.reset_index()

    # Jointure via table intermédiaire des ids
    join_idx = pandas.DataFrame(
        {
            "right": match_index.flatten(),
            "join_measure": match_measure.flatten(),
        },
        index=[i for i in range(match_index.shape[0])
               for _ in range(match_index.shape[1])],
    )
    return (
        pandas.concat([left, join_idx], axis=1)
        .merge(right.drop("geometry", axis=1), how="left",
               left_on="right", right_index=True)
        .drop("right", axis=1)
        .set_index(lindex_names)
    )
