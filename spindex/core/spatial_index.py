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
Spatial Index data structure and algorithms.

The base data structure is the class :class:`IndexTree`.
Different algorithms for building such a tree are provided via subclasses of
:class:`IndexTree`.
'''
import pdb
import copy
import heapq
import time
import warnings

import sklearn.cluster
import toolz

import spindex.core.enclosing_geometry

# ====================  BGH Data Structure  =============================

# Bounding Geometry Tree (aka bounding volumes hierarchy)
#
# Tree structure: node with list of children
# Node stores bounding geometries, possibly a node_obj and its children.
# For internal node, children is a list of children trees.
# For a leaf node, it is a mapping to bounding_geometries.
# Implements state pattern with [empty, non-empty] states.
# Implements visitor pattern


class BoundingGeometryTree():
    """
    Bounding Volumes Hierarchy data-structure.

    Abstract tree data-structure for bounding volumes hierarchies.  Each node
    has a bounding geometry from :mod:`enclosing_geometry` and children. For
    internal nodes, children is a list of children `BoundingGeometryTree`. For
    leaf nodes, children is a mapping of bounding geometries.

    The class implements the state and visitor patterns. To add a visitor,
    create a class with at least two methods: visit_empty and visit.

    Attributes
    ----------
    bounds: bounding geometry
    node_obj: object
        Optional node object, e.g. centers.
    children: list or mapping
        List of children trees or leaf ids -> bounding geometry.
    isleaf: boolean
    """
    __slots__ = ["bounds", "node_obj", "children", "isleaf"]

    def __init__(self):
        self.bounds = None
        self.node_obj = None
        self.children = None
        self.isleaf = False

    @property
    def isempty(self):
        '''Returns True if the tree is empty.'''
        return self.bounds is None

    # State: empty -> NonEmpty
    def create_root(self, bgeom):
        '''Change state from empty to non-empty tree. Creates the root node,
        making its bounds equal to `bgeom`.'''
        if self.isempty:
            self.bounds = bgeom

    # State: NonEmpty -> empty
    def delete(self):
        '''Changes state from non-empty to empty tree. Resets attributes.'''
        self.bounds = None
        self.node_obj = None
        self.children = None
        self.isleaf = False

    def insert_child(self, bgtree):
        '''Insert the single child `bgtree`.'''
        self.children.append(bgtree)

    def delete_child(self, idx=-1):
        '''Removes the child at position `idx`.'''
        if self.isempty or self.isleaf:
            warnings.warn("Cannot delete child from a leaf or empty tree. "
                          "No action taken.")
        else:
            del self.children[idx]
            if len(self.children) == 0:
                self.children = None
                self.isleaf = True

    def accept(self, visitor, *args, **kwargs):
        '''Accept `visitor` to operate on the structure.'''
        if self.isempty:
            return visitor.visit_empty(self, *args, **kwargs)
        else:
            return visitor.visit(self, *args, **kwargs)


# ====================  BGH Builders  =============================


class DKMeansBulkInsert():
    '''
    Divisive KMeans BGH-builder.

    DKMeansBulkInsert is a top-down hierarchical clustering. Division is done
    via KMeans clustering.

    Attributes
    ----------
    max_children: int
        Max number of children for internal node.
    max_leaves: int
        Max number of leaves in a left object.
    n_jobs: int
        Number of jobs to run in parallel.
    '''
    def __init__(self, max_children=6, max_leaves=64, n_jobs=1):
        self.max_children = max_children
        self.max_leaves = max_leaves
        self.n_jobs = n_jobs

    @staticmethod
    def _cluster(rgeoms, n_clusters=6, n_init=3, n_jobs=1):
        cluster_model = sklearn.cluster.KMeans(
            n_clusters=n_clusters,
            random_state=0,
            n_init=n_init,
            n_jobs=n_jobs,
        )
        centers = [tuple(g.centroid.coords)[0] for _, g in rgeoms.items()]
        fitted = cluster_model.fit(centers)
        clust = {k: v for k, v in zip(rgeoms.keys(), fitted.labels_)}
        return (clust, fitted.cluster_centers_)

    # For parallelization purposes
    def _index_task(self, bgeoms):
        visitor = copy.copy(self)
        visitor.n_jobs = 1
        tree = BoundingGeometryTree()
        tree.accept(visitor, bgeoms)
        return tree

    def visit_empty(self, bgtree, bgeoms):
        bvalues = [b for _, b in bgeoms.items()]
        bgtree.create_root(spindex.core.enclosing_geometry.bound_all(bvalues))
        # insert_node_obj for root?
        accept_stack = [(bgtree, bgeoms)]
        while len(accept_stack) > 0:
            tree, bound_geos = accept_stack.pop(0)
            if len(bound_geos) <= self.max_leaves:
                tree.children = bound_geos
                tree.isleaf = True
            else:
                tree.children = []
                clust, centers = self._cluster(bound_geos,
                                               n_clusters=self.max_children)
                # Groupby the geometries by cluster.
                groups = toolz.valmap(
                    dict,
                    toolz.groupby(lambda x: clust[x[0]], bound_geos.items())
                )
                # Parallel or not
                if self.n_jobs == 1:
                    for clust_id, grp in groups.items():
                        bvalues = [b for _, b in grp.items()]
                        bgeom = spindex.core.enclosing_geometry.bound_all(
                            bvalues)
                        child_tree = BoundingGeometryTree()
                        child_tree.bounds = bgeom
                        child_tree.node_obj = centers[clust_id]
                        tree.insert_child(child_tree)
                        accept_stack.append((child_tree, grp))
                else:
                    import multiprocessing
                    with multiprocessing.Pool(self.n_jobs) as pool:
                        child_trees = pool.map(
                            self._index_task,
                            [g for _, g in groups.items()]
                        )
                    bgtree.children = child_trees

    def visit(bgtree, bgeoms):
        # bulk update
        raise NotImplementedError


# ====================  Visitors  =============================
# Visitor not in the sense of visitor pattern, but in the sense of visiting the
# data-structure.


class ApproxNearest():
    '''
    Approximate nearest neighbours generator BoundingGeometryTree visitor.
    '''
    @staticmethod
    def visit_empty(bgtree, bgeom):
        return

    @staticmethod
    def visit(bgtree, bgeom):
        nodes_ref = [bgtree]
        n_nodes = 1
        nodes_heap = [(bgeom.mindist(bgtree.bounds), 0, True)]
        # Iterate through the tree starting with the heap's min element.
        while nodes_heap:
            mindist, idx, is_node = heapq.heappop(nodes_heap)
            node = nodes_ref[idx] if is_node else None
            if is_node and not node.isleaf:
                for child in node.children:
                    heapq.heappush(
                        nodes_heap,
                        (bgeom.mindist(child.bounds), n_nodes, True)
                    )
                    nodes_ref.append(child)
                    n_nodes += 1
            elif is_node and node.isleaf:
                for idx, obj in node.children.items():
                    heapq.heappush(nodes_heap,
                                   (bgeom.mindist(obj), idx, False))
            else:
                yield (idx, mindist)


class Intersection():
    """
    Predicate based-query BoundingGeometryTree visitor.

    Attributes
    ----------
    op: one of ['within', 'contains', 'intersects']
        Which predicate to use for the query.
    """
    def __init__(self, op='intersects'):
        self.op=op

    def visit_empty(self, bgtree, bgeom):
        return

    def visit(self, bgtree, bgeom):
        predicate = (
            bgeom.within if self.op == 'within' else
            bgeom.contains if self.op == 'contains' else
            bgeom.intersects
        )
        node_stack = [bgtree]
        while len(node_stack) > 0:
            tree = node_stack.pop(0)
            if not predicate(tree.bounds):
                continue
            if tree.isleaf:
                for i, obj in tree.children.items():
                    if predicate(obj):
                        yield i
            else:
                node_stack.extend(tree.children)
