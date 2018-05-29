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
Bounding volume classes.

Spatial indexes are built using simplifications of geometrical shapes. The
simplified geometries are shapes enclosing the given geometry and are called
bounding volumes.

Classically, axis-aligned minimum bounding boxes (AAMBR) are used.
This choice is in our opinion mainly due to its simplicity.
Equally simple from the point of vue of both space and computation efficiency
is the smallest bounding sphere (SBS). The latter exhibit a few
advantages over AAMBR, and is also implemented in this module.
'''
import abc
import pdb

import numpy
import shapely.geometry

import spindex.externals.smallest_enclosing_circle as secircle


# Dispatching bound_all to its class method
def bound_all(bgeoms):
    # Get the class of first value in bgeoms.
    cls = type(next(iter(bgeoms)))
    return cls.merge(bgeoms)


class Enclosing_Geometry(abc.ABC):
    """
    Abstract interface for bounding volume class.

    A bounding volume is a geometry enclosing entirely an underlying geometry.
    Examples: axis-aligned minimum bounding rectangle (AAMBR) and smallest
    bounding sphere (SBS).
    """
    __slots__ = ()

    @classmethod
    @abc.abstractmethod
    def merge(cls, collection):
        """
        Returns a bounding volume of a collection of bounding volumes.
        """
        pass

    @abc.abstractmethod
    def intersects(self, other):
        """
        Returns True if `self` intersects the bounding volume `other`.
        """
        pass

    @abc.abstractmethod
    def centroid(self):
        """Returns `self`'s barycenter."""
        pass

    @abc.abstractmethod
    def mindist(self, other):
        """
        Returns a lower bound on the distance between `self`'s and `other`'s
        possible underlying geometries.
        """
        pass

    @abc.abstractmethod
    def maxdist(self, other):
        """
        Returns an upper bound on the distance between `self`'s and `other`'s
        possible underlying geometries.
        """
        pass


class Rect(shapely.geometry.Polygon, Enclosing_Geometry):
    '''Axis-aligned minimum bounding rectangle.'''
    @staticmethod
    def merge(collection):
        arr = numpy.array([r.bounds for r in collection])
        mins = arr[:, :2].min(0)
        maxs = arr[:, 2:].max(0)
        return Rect(*mins, *maxs)

    def __init__(self, *args):
        if not args:
            minx = numpy.nan
            miny = numpy.nan
            minx = numpy.nan
            maxy = numpy.nan
        elif len(args) == 1:
            minx, miny, maxx, maxy = args[0]
        else:
            minx, miny, maxx, maxy = args
        super(Rect, self).__init__([(minx, miny), (maxx, miny),
                                    (maxx, maxy), (minx, maxy),
                                    (minx, miny)])

    def __repr__(self):
        return "Rect(minx={}, miny={}, maxx={}, maxy={})".format(*self.bounds)

    def mindist(self, to_other):
        return self.distance(to_other)

    def maxdist(self, to_other):
        bthis = self.bounds
        bthat = to_other.bounds
        xmaxmin = abs(bthis[2] - bthat[0])
        xmaxmax = abs(bthis[2] - bthat[2])
        xminmin = abs(bthis[0] - bthat[0])
        xminmax = abs(bthis[0] - bthat[2])
        ymaxmin = abs(bthis[3] - bthat[1])
        ymaxmax = abs(bthis[3] - bthat[3])
        yminmin = abs(bthis[1] - bthat[1])
        yminmax = abs(bthis[1] - bthat[3])
        return numpy.sqrt(min(
            max(xmaxmin, xminmax)**2
            + min(ymaxmin, ymaxmax, yminmin, yminmax)**2,
            max(ymaxmin, yminmax)**2
            + min(xmaxmin, xmaxmax, xminmin, xminmax)**2,
            min(max(ymaxmin, ymaxmax), max(yminmin, yminmax))**2
            + min(max(xminmax, xmaxmax), max(xminmin, xmaxmin))**2,
            min(max(xmaxmin, xmaxmax), max(xminmin, xminmax))**2
            + min(max(yminmax, ymaxmax), max(yminmin, ymaxmin))**2,
        ))


class Sphere(Enclosing_Geometry):
    '''Smallest bounding sphere.'''
    __slots__ = ('x', 'y', 'r')

    def __init__(self, *args, buf=10**-6):
        # Overloading
        # 1st case: geom + tolerance
        if hasattr(args[0], "coords"):
            self.x, self.y, self.r = secircle.make_circle(args[0].coords)
        elif hasattr(args[0], "exterior"):
            self.x, self.y, self.r = secircle.make_circle(
                args[0].exterior.coords)
        # 2nd case: triple (x, y) for the center and r for the radius.
        else:
            self.x = args[0]
            self.y = args[1]
            self.r = args[2]
        self.r += buf

    def _center_dist_sq(self, point):
        return (self.x - point[0])**2 + (self.y-point[1])**2

    def intersects(self, other):
        return self._center_dist_sq((other.x, other.y)) < (self.r + other.r)**2

    def center(self):
        return [self.x, self.y]

    def merge(self, others):
        raise NotImplementedError

    def mindist(self, to_other):
        return max(0, numpy.sqrt(self._center_dist_sq(to_other.center))
                      - self.r - to_other.r)

    def maxdist(self, to_other):
        return numpy.sqrt(self._center_dist_sq(to_other.center)
                          + (self.r + to_other.r)**2)

    def to_array(self):
        return array.array('d', (self.x, self.y, self.r))
