"""
STR-Tree packing algorithm

Sort-Tile-Recurse tree packing algorithm is simple and efficient.
This module implements it with a highly efficient data structure for speed.
"""
import math

import numpy
import numpy.ma

from . import envelope
from . import tree


# Works for 2d arrays, concat along first axis.
# Work on a more general function.
def _concat(gen, shapes, fill_value=0):
    """
    Concatenation of arrays with known shapes with preallocated memory.

    Args:
        gen (array generator): Generator of arrays.
        shapes (array): Shapes of the generated arrays concatenated along the
            first axis. Arrays' dimensions must be the same and their shapes
            must be known in advance.
        fill_value (object): Default value to fill arrays with smaller shapes.

    Returns:
        array: the concatenation of the arrays.
    """
    right_ix = shapes[:, 0].cumsum()
    left_ix = numpy.concatenate([[0], right_ix[:-1]])
    shape = (right_ix[-1], shapes[:, 1].max())
    res = numpy.ma.array(
        numpy.full(shape=shape, fill_value=fill_value),
        mask=numpy.ones(shape=shape),
        fill_value=fill_value,
    )
    for arr, l, r, j in zip(gen, left_ix, right_ix, shapes[:, 1]):
        res[l:r, :j] = arr.reshape(r-l, j)
    return res


def sort_tile_recurse(egeoms, page_size=16, max_top_size=1):
    """

    Parameters:
        egeoms (Envelope): single geometry envelopes.

    Returns:
        IndexTree: A R-Tree index of the envelopes.
    """
    def sort_tile_one(xs, nb_tiles):
        """Sort and tile along first coordinate."""
        argx = numpy.argsort(xs)
        q, r = divmod(len(xs), nb_tiles)
        if r == 0:
            return numpy.ma.array(argx.reshape(nb_tiles, q))
        split = (q+1) * r
        return _concat(
            gen=(argx[:split], argx[split:]),
            shapes=numpy.array([(r, q+1), (nb_tiles-r, q)]),
            fill_value=0,
        )

    def get_shape(nobs, ndims):
        """Shape of children after grouping along one coordinate."""
        nb_tiles = math.ceil((nobs/page_size)**(1/ndims))
        return (nb_tiles, math.ceil(nobs/nb_tiles))

    def sort_tile(coords):
        nb_tiles, _ = get_shape(*coords.shape)
        splits = sort_tile_one(coords[:, 0], nb_tiles)
        if coords.shape[1] == 1:
            return splits
        else:
            array_gen = (s.compressed()[sort_tile(coords[s.compressed(), 1:])]
                         for s in splits)
            shapes = numpy.array([
                get_shape(s.count(), coords.shape[1] - 1)
                for s in splits
            ])
            return _concat(array_gen, shapes, fill_value=0)

    # Initialise the leaves
    envel = [egeoms]
    children = [numpy.ma.arange(len(egeoms)).reshape(-1, 1)]
    while len(envel[-1]) > max_top_size:
        children.append(sort_tile(coords=envel[-1].centers))
        envel.append(envel[-1].mergeby(children[-1]))
    return tree.BVH(envel[::-1], children[::-1])


    # Implementation with lists
    # def sort_tile_one(xs, nb_tiles):
    #     argx = numpy.argsort(xs)
    #     q, r = divmod(len(xs), nb_tiles)
    #     split_idx = numpy.cumsum([q + int(x < r) for x in range(nb_tiles - 1)])
    #     return numpy.split()
    #
    # def sort_tile(coords):
    #     nb_tiles = math.ceil((coords.shape[0]/page_size)**(1/coords.shape[1]))
    #     splits = sort_tile_one(coords[:, 0], nb_tiles)
    #     if coords.shape[1] == 1:
    #         return splits
    #     else:
    #         return [sp[x] for sp in splits for x in sort_tile(coords[sp, 1:])]
