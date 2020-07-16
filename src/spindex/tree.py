"""
TODO
"""
import pdb

import collections

import numpy


class BVH():
    """
    Highly efficient R-Tree-like structure using arrays.

    The structure of a BVH is like a R-Tree, but the initial level is not
    constrained to contain a single node (the root). Instead, it could be
    cut-off at the higher levels, leading to a balanced forest instead.

    A more important difference comparing to a R-Tree is that arrays are used
    instead of the classically doubly linked list of siblings at each level.

    Note:
        The name BVH stands for Bounding Volume Hierarchy, which is both more
        descriptive and general than R-Tree. The name comes from collision
        detection in computer graphics. In a bounding volume hierarchy, the
        bounding geometries (a.k.a envelopes) need not be axis-aligned
        rectangles, nor even rectangles at all. We also add that a hierarchy
        need not be a single tree.

    Args:
        envelopes: value of the envelopes attribute.
        children: value of the children attribute.

    Attributes:
        envelopes (list of EnvelopeVect): nodes' envelopes per tree level.
            This replaces the doubly linked list of siblings by arrays.
        children (list of 1d-int-array): indices in the next levels envelopes
            defining this node's children.
    """
    valid_predicates = ("intersects", "contains", "within",
                        "overlaps", "crosses", "touches")

    def __init__(self, envelopes, children):
        self.envelopes = envelopes
        self.children = children

    @property
    def width(self):
        """Number of nodes at top level."""
        if self.is_empty:
            return 0
        return len(self.envelopes[0])

    @property
    def depth(self):
        """Depth of the tree."""
        return len(self.envelopes)

    @property
    def is_empty(self):
        """Boolean: Is the tree empty?"""
        return self.depth == 0

    def __len__(self):
        """Returns the number of leaves."""
        return len(self.envelopes[-1])

    def search(self, obj, search_func, sparse=False):
        """
        Bulk branch-and-bound search query with given predicate function.
        """
        # Cascading through levels to get matching indexes.
        paths = [QueryPath(
            query=numpy.arange(len(obj)),
            target=numpy.arange(self.width),
        )]
        empty_paths = []
        for envel, children in zip(self.envelopes, self.children):
            next_paths = []
            for path in paths:
                pred = search_func(obj[path.query], envel[path.target])
                # Early stopping for non-matching query geometries
                is_nonempty = pred.any(axis=1)
                pred = pred[is_nonempty]
                empty_paths.append(path.query[~is_nonempty])
                query = path.query[is_nonempty]
                if is_nonempty.any():
                    next_paths.extend([
                        QueryPath(
                            query=query[grp],
                            target=(
                                children[path.target[pred[grp[0], :]], :]
                                .compressed()
                            ),
                        ) for grp in groupby(pred, axis=1)
                    ])
            paths = next_paths
        if kind == 'left':
            paths.append(QueryPath(
                query=numpy.concatenate(empty_paths),
                target=numpy.array([], dtype=int),
            ))
        if sparse:
            return paths
        return _sparse_to_full(paths, len(obj))

    def query(self, obj, predicate, kind='inner', sparse=False):
        """
        Args:
            obj (EnvelopeVect): query objects
            predicate (str): One of {}
            sparse (bool, optional): If True, returns a sparser representation
                of the resulting indexes. Defaults to False.

        Returns:
            If sparse is False, returns a list with an array of matching
            indices for each element of obj.
            If spare is True, returns a list of 2-tuples of arrays of matching
            indices.
        """.format(self.valid_predicates)
        if predicate not in self.valid_predicates:
            raise ValueError(
                "Invalid predicate {}: must be one of {}"
                .format(predicate, self.valid_predicates)
            )

        def search_func(obj, envel):
            return getattr(obj, predicate)(envel)

        return self.search(obj, search_func, kind=kind, sparse=sparse)

    def nearest(self, obj, knn=1, sparse=False):
        """
        TODO
        """
        def search_func(obj, envel):
            lower_bounds, upper_bounds = obj.bound_dist(envel)
            if upper_bounds.shape[1] <= knn:
                kth_upper_bounds = upper_bounds.max(axis=1)
            else:
                kth_upper_bounds = numpy.apply_along_axis(
                    lambda arr: numpy.partition(arr.flatten(), kth=knn)[knn],
                    axis=1,
                    arr=upper_bounds,
                )
            return (lower_bounds.T <= kth_upper_bounds).T  # Bdcast on 1st dim

        return self.search(obj, search_func, sparse=sparse)


QueryPath = collections.namedtuple("QueryPath", "query target")
# Path corresponds to a search path consisting of indexes in both query and
# target geoms


def _sparse_to_full(paths, length):
    # Resulting valid paths are reordered in original obj order.
    result = [None]*length
    for path in paths:
        for i in path.query:
            result[i] = path.target
    return result


def groupby(arr, axis=1):
    """
    Groupby array on values along axis _axis_.

    The operation uses the hash method _tobytes()_ to quickly compare elements.

    Args:
        arr (array): Array to group by values on an axis.
        axis (int, optional): Axis along which to group values.
            Defaults to 1.
    """
    # Hash values using tobytes(), sort them, and performs a unix groupby
    # (via the uniq command)
    hashes = (arr * 2**numpy.arange(arr.shape[1])).sum(axis=1)
    arg = numpy.argsort(hashes)
    _, ind = numpy.unique(hashes[arg], return_index=True)
    return numpy.split(arg, ind[1:])
