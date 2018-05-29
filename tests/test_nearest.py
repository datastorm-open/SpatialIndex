import pdb
import pytest

import shapely.geometry
import geopandas
import numpy

import spindex.core.enclosing_geometry as encl
import spindex.core.spatial_index as sidx
import spindex.core.spatial_joins as sjoin


@pytest.fixture
def left_df():
    geoms = [shapely.geometry.LineString([(0, 0), (1, 1)]),
             shapely.geometry.LineString([(3, 0), (-2, 2)])]
    attributes = ['z', 'y']
    res = geopandas.GeoDataFrame({'geometry': geoms, 'attr': attributes})
    res.index.name = 'left_id'
    return res


@pytest.fixture
def right_df():
    geoms = [shapely.geometry.LineString([(0, 1), (1, 2)]),
             shapely.geometry.LineString([(1, 0), (2, 2)]),
             shapely.geometry.LineString([(1, 1.1), (0, 2)]),
             shapely.geometry.LineString([(0, 0), (2, 0), (2, 2),
                                          (0, 2), (0, 0)]),
             shapely.geometry.LineString([(-1.3, 0), (-1, 2)]),
             shapely.geometry.LineString([(-1.3, 0), (-1, 2)]),
             shapely.geometry.LineString([(-1, -1), (-0.5, 0)]),
             shapely.geometry.LineString([(-10, 0), (-1, 5)]),
             shapely.geometry.LineString([(0.5, -0.5), (0, -1), (-1, 4)]),
             shapely.geometry.LineString([(4, 0.6), (-3, 0.5)])]
    attributes = ['A', 'B', 'A', 'B', 'B', 'C', 'D', 'C', 'E', 'E']
    res = geopandas.GeoDataFrame({'geometry': geoms, 'attr': attributes})
    res.index.name = 'right_id'
    return res


@pytest.fixture
def right_sindex(right_df):
    bounds = right_df.geometry.apply(lambda g: encl.Rect(g.bounds))
    tree = sidx.BoundingGeometryTree()
    tree.accept(sidx.DKMeansBulkInsert(max_children=16, max_leaves=64),
                bounds)
    return tree


def pairwise_query(left_geom, right_df):
    distances = right_df.geometry.apply(lambda g: left_geom.distance(g))
    return [(i, d) for i, d in distances.sort_values().items()]


def test_query_one_neighbour(left_df, right_df, right_sindex):
    for l_geom in left_df.geometry:
        res = sjoin.nearest(l_geom, right_df.geometry, right_sindex)
        pres = pairwise_query(l_geom, right_df)
        assert abs(l_geom.distance(right_df.loc[res[0][0], 'geometry'])
                   - l_geom.distance(right_df.loc[pres[0][0], 'geometry'])) < 10**-6


def test_query_three_neighbours(left_df, right_df, right_sindex):
    for l_geom in left_df.geometry:
        res = sjoin.nearest(l_geom, right_df.geometry, right_sindex, n_neighbours=3)
        pres = pairwise_query(l_geom, right_df)
        assert all([abs(l_geom.distance(g1) - l_geom.distance(g2)) < 10**-6
                    for g1, g2 in zip(right_df.loc[[x[0] for x in res],
                                                   'geometry'],
                                      right_df.loc[[x[0] for x in pres],
                                                   'geometry'])])


def test_core_join_one_neighbour(left_df, right_df):
    l_geom = left_df.geometry
    r_geom = right_df.geometry
    res = sjoin.nearest_join(l_geom, r_geom)
    pres = [(i, j, d) for i, l in left_df.geometry.items()
            for j, d in pairwise_query(l, right_df)[:1]]

    assert all([all(r == p) for r, p in zip(res, pres)])


# def test_core_join_one_neighbour(left_df, right_df):
#     l_df = left_df.reset_index()
#     r_df = right_df.reset_index()
#     res = sjoin.nearest(l_df, r_df)
#     pres = [pairwise_query(l, right_df)[0] for l in left_df.geometry]
# 
#     assert isinstance(res, geopandas.GeoDataFrame)
#     assert all(res['right_id'] == pres)
#     assert all(res['left_id'] == left_df.index)
#     assert all(res['attr_left'] == ['z', 'y'])
#     assert all(res['attr_right'] == ['z', 'y'])
#     assert all([this.equals(that) for this, that
#                 in zip(res.geometry, left_df.geometry)])
# 
