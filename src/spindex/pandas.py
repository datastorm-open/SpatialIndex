"""
Module wrapping pandas DataFrames.
"""
import pdb
import numpy
import pandas

import spindex.core.data_providers
import spindex.core.spatial_joins


class GeoFrame():
    """
    DataFrame class with a distinguish `GIShapes` class for geometries.

    Attributes
    ----------
    geometry: spindx GIShapes
    attributes: pandas DataFrame
    """

    __slot__ = ('geometry', 'attributes')

    def __init__(self, dframe):
        self.geometry = spindex.core.data_providers.GIShapes(
            dframe['geometry'])
        self.geometry.create_index()
        self.attributes = dframe.drop('geometry', axis=1)

    def to_pandas(self):
        return pandas.concat([self.geometry.data.to_frame(), self.attributes],
                             axis=1)


def st_join(left, right, how='left', op='knn', include_measure=True, **kwargs):
    """
    Spatial joins for DataFrames.

    `st_join` performs a spatial join where the predicate is specified by the
    arguments `how` and `op`. For example, it can perform a left true nearest-
    neighbours join.

    The DataFrames must have a column named 'geometry' consisting of Shapely's
    geometries or the like. The geometries can be heterogeneous.

    Parameters
    ----------
    left: pandas DataFrame
    right: GIFrame or pandas DataFrame
    how: str (default 'left')
        'left' or 'inner' for the type of join. Left joins are implemented so
        far.
    op: str (default 'knn')
        1. 'knn': true nearest-neighbours join. For each left geometry, it
            finds the actual `n_neighbours`-nearest neighbours in the metric
            given by the distance between shapes.
    include_measure: bool (default True)
        1. 'knn': should the distance to the neighbours be included in the
                  result.
    kwargs:
        keyword arguments depending on `op`:
            1. 'knn': n_neighbours, int (default 1).

    Returns
    -------
    pandas DataFrame
    """
    if isinstance(right, pandas.DataFrame):
        right = GIFrame(right)
    if isinstance(left, GIFrame):
        left = left.to_pandas()
    elif isinstance(left, pandas.DataFrame):
        geoms = left['geometry']
    elif isinstance(left, pandas.Series):
        geoms = left
    else:
        raise ValueError("Unrecognized type for left frame.")

    join = list(spindex.core.spatial_joins.st_join(geoms, right.geometry,
                                                   how, op, **kwargs))
    join = numpy.array(join)
    if op == 'knn':
        return _wrap_knn(left, right, join, include_measure, **kwargs)
    else:
        raise NotImplementedError("Joins of type {} are not implemented"
                                  .format(op))


def _wrap_knn(left, right, join, include_measure, **kwargs):
    """
    Reshapes the join result into the appropriate pandas DataFrame.

    Parameters
    ----------
    left: pandas DataFrame
    right: GIFrame
    join: numpy array
        Result of the join.

    Returns
    -------
    pandas DataFrame
    """
    n_neighbours = kwargs.get("n_neighbours", 1)
    pdb.set_trace()
    join = join.reshape(-1, join.shape[2])
    if include_measure:
        # join.shape is len(left) x n_neighbours x 2
        join = join.reshape(-1, 2)
    else:
        # join.shape is len(left) x n_neighbours x 1
        join = join[:, :, 0].reshape(-1, 1)
    left_index = numpy.repeat(left.index.values, n_neighbours)
    res = pandas.concat(
        [left.loc[left_index].reset_index(drop=True),
         right.attributes.loc[join[:, 0]].reset_index(drop=True)],
        axis=1
    )
    if include_measure:
        res = pandas.concat([res, pandas.DataFrame(join[:, 1],
                                                   columns=['knn_distance_'])],
                            axis=1)
    return res
