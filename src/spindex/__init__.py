"""
Highly efficient spatial indexing and querying.

Classically, R-trees are implemented as a container of node objects.  Siblings
are related through a doubly-linked list, and nodes store further pointers to
parent and child nodes.

Our implementation differs in that doubly-linked lists are replaced by arrays,
allowing for most operations to be vectorized.  This leads to a large increase
in spped.  As it is written entirely in python, it is also more extensible.


For example, sort-tile-recurse packing algorithm is used to construct a
R-tree, but it is possible to extend the library with a different algorithm.
"""
from .envelope import Envelopes, AAMBRs  # noqua: F401
from .build import sort_tile_recurse  # noqa: F401
from .tree import BVH  # noqa: F401

__version__ = "0.2.0"
