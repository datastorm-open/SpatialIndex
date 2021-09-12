import geopandas
import shapely.geometry

import spindex.joins as joins

geoms = [
    shapely.geometry.LineString([(0, 0), (1, 1)]),
    shapely.geometry.LineString([(0, 0), (1, 0)]),
    shapely.geometry.LineString([(0, 0), (0, 1)]),
    shapely.geometry.LineString([(1, 0), (2, 0), (2, -1)]),
    shapely.geometry.LineString([(0, 1), (0.2, 1.5), (1, 2)]),
    shapely.geometry.LineString([(1, 2), (1, 3)]),
    shapely.geometry.LineString([(1, 3), (2, 3)]),
    shapely.geometry.LineString([(1, 3), (1, 4)]),
    shapely.geometry.LineString([(0, 2), (1, 2)]),
    shapely.geometry.LineString([(0, 2), (0, 4), (-1, 4)]),
    shapely.geometry.LineString([(1, 2), (2, 2)]),
    shapely.geometry.LineString([(1, 0), (2, 1)]),
    shapely.geometry.LineString([(2, 1), (2, 2)]),
    shapely.geometry.LineString([(2, 1), (3, 1), (4, 1)]),
    shapely.geometry.LineString([(4, 1), (5, 2)]),
]
frame = geopandas.GeoDataFrame({
    'feat': list(range(5, 20)),
    'geometry': geoms,
})
geoms = frame.geometry.values
pgeoms = joins.prepare(geoms)

pts = geopandas.GeoDataFrame({'geometry':[
    shapely.geometry.Point(1, 1.5),
    shapely.geometry.Point(2, 2.2),
    shapely.geometry.Point(4, 0),
]})
areas = geopandas.GeoDataFrame({'geometry': [
    shapely.geometry.Polygon([(1.2, 2.8), (1.2, 1.8), (0.8, 0.6), (0.6, 0.8), (1.2, 2.8)]),
    shapely.geometry.Polygon([(2.8, 1.2), (1.8, 1.2), (0.6, 0.8), (0.8, 0.6), (2.8, 1.2)]),
]})
