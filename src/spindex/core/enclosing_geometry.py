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
import numpy
import spindex.externals.smallest_enclosing_circle as secircle


class Enclosing_Geometry(abc.ABC):
    """ 
    Abstract interface for bounding volume class.

    A bounding volume is a geometry enclosing entirely an underlying geometry.
    Examples: axis-aligned minimum bounding rectangle (AAMBR) and smallest
    bounding sphere (SBS).
    """
    __slots__ = ()

    @abc.abstractmethod
    def intersects(self, other):
        """
        Returns True if `self` intersects the bounding volume `other`.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def merge(cls, collection):
        """
        Returns a bounding volume of a collection of bounding volumes.
        """
        pass

    @abc.abstractmethod
    def center(self):
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


class Rect(Enclosing_Geometry):
    '''Axis-aligned minimum bounding rectangle.'''
    __slots__ = ('minx', 'miny', 'maxx', 'maxy', 'buf')

    @staticmethod
    def merge(collection):
        return Rect(min([r.minx for r in collection]),
                    min([r.miny for r in collection]),
                    max([r.maxx for r in collection]),
                    max([r.maxy for r in collection]),
                    )

    def __init__(self, *args, buf=10**-6):
        if not args:
            self.minx = numpy.nan
            self.miny = numpy.nan
            self.maxx = numpy.nan
            self.maxy = numpy.nan
        elif len(args) == 1:
            self.minx, self.miny, self.maxx, self.maxy = args[0].bounds
        else:
            self.minx = float(args[0])
            self.miny = float(args[1])
            self.maxx = float(args[2])
            self.maxy = float(args[3])

        self.minx -= buf
        self.miny -= buf
        self.maxx += buf
        self.maxy += buf
        self.buf = buf

    def __repr__(self):
        return "Rect(minx={}, miny={}, maxx={}, maxy={})".format(
                    self.minx, self.miny, self.maxx, self.maxy)

    def intersects(self, other):
        minx = max(self.minx, other.minx)
        miny = max(self.miny, other.miny)
        maxx = min(self.maxx, other.maxx)
        maxy = min(self.maxy, other.maxy)
        if minx < maxx and miny < maxy:
            return True  # Rect(minx, miny, maxx, maxy)
        else:
            return False  # Rect()

    def center(self):
        return [(self.minx + self.maxx)/2, (self.miny + self.maxy)/2]

    def mindist(self, to_other):
        x = max(0, self.minx - to_other.maxx, to_other.minx - self.maxx) 
        y = max(0, self.miny - to_other.maxy, to_other.miny - self.maxy) 
        return numpy.sqrt(x**2+y**2)

    def maxdist(self, to_other):
        Xmaxmin = abs(self.maxx - to_other.minx)
        Xmaxmax = abs(self.maxx - to_other.maxx)
        Xminmin = abs(self.minx - to_other.minx)
        Xminmax = abs(self.minx - to_other.maxx)
        Ymaxmin = abs(self.maxy - to_other.miny)
        Ymaxmax = abs(self.maxy - to_other.maxy)
        Yminmin = abs(self.miny - to_other.miny)
        Yminmax = abs(self.miny - to_other.maxy)
        return numpy.sqrt(min(
            max(Xmaxmin, Xminmax)**2 + min(Ymaxmin, Ymaxmax, Yminmin, Yminmax)**2,
            max(Ymaxmin, Yminmax)**2 + min(Xmaxmin, Xmaxmax, Xminmin, Xminmax)**2,
            min(max(Ymaxmin, Ymaxmax), max(Yminmin, Yminmax))**2 + min(max(Xminmax, Xmaxmax), max(Xminmin, Xmaxmin))**2,
            min(max(Xmaxmin, Xmaxmax), max(Xminmin, Xminmax))**2 + min(max(Yminmax, Ymaxmax), max(Yminmin, Ymaxmin))**2,
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
