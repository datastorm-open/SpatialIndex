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
import time
import collections
import heapq
import toolz
import sklearn.cluster


# ====================  IndexTree Data Structure  =============================

# The IndexTree pertains only to the data structure itself: definition,
# iteration, and simple inserts. Since there are many algorithms to build an
# IndexTree, they are separated from the data structure.
#
# The data model for the tree is given by the following specifications:
#   1. Nodes are stored in a 1d-buffer indexed by non-negative integers.
#   1. The root node has index 0.
#   1. There are 2 types of nodes:
#          a. internal nodes that have node children.
#          a. leaf nodes that point to a buffer of leaf objects.
#   1. A node contains at least a tuple of
#          a. the index of its first child or of its leaf buffer.
#          a. the index of its first sibling.
#          a. a bool of whether it is a leaf node (set to false).
#   1. A non-positive sibling index indicates that there is no sibling
#      (should happen only for the root node).
#   1. To each node corresponds an enclosing geometry containing all shapes
#      associated to the node.
#   1. The enclosing geometries are stored in a 1d-buffer parallel to the
#      nodes.
#   1. Leaf buffers contain a subcollection of keys to the shapes.
#   1. Leaf buffers are stored in a 1d-buffer indexed by non-negative
#      integers.

BaseNode = collections.namedtuple('BaseNode_', 'child sibling isleaf')


class IndexTree():
    """Data structure of a spatial index tree."""
    def __init__(self, max_children=16, max_leaves=64):
        self.max_children = max_children
        self.max_leaves = max_leaves
        self._clear()

    def _clear(self):
        self._enclose_class = None
        self.node_count = 0
        self.leaf_count = 0
        self.nodes = []
        self.preps = []
        self.leaves = []

    @property
    def isempty(self):
        return self.node_count == 0

    def root(self):
        if self.isempty:
            raise ValueError("Index tree is empty")
        return 0

    def siblings(self, idx=0):
        node = self.nodes[idx]
        for _ in range(self.max_children):
            if node.sibling == 0:
                return
            yield node.sibling
            node = self.nodes[node.sibling]

    def children(self, idx=0):
        node = self.nodes[idx]
        if node.isleaf:
            return
        yield node.child
        yield from self.siblings(node.child)

    def leaf(self, idx=0):
        leaf = self.leaves[idx]
        yield from leaf.items()

    def insert_node(self, node, prep):
        self.nodes.append(node)
        self.preps.append(prep)
        self.node_count += 1

    def insert_leaf(self, pgeoms):
        self.leaves.append(pgeoms)
        self.leaf_count += 1

    def approx_intersects(self, prep, idx=0):
        node = self.nodes[idx]
        if not prep.intersects(self.preps[idx]):
            return
        if node.isleaf:
            yield from (idx for idx, oprep in self.leaf(node.child)
                        if prep.intersects(oprep))
        else:
            for child in self.children(node):
                yield from self.approx_intersects(prep, child)

    def approx_nearest(self, refprep, idx=0):
        if self.isempty:
            return

        INT_NODE = 0
        LEAF_NODE = 1
        LEAF_OBJ = 2

        nodes_heap = []
        typ = int(self.nodes[idx].isleaf)
        prep = self.preps[idx]
        heapq.heappush(nodes_heap, (refprep.mindist(prep), idx, typ))

        # Iterate through the tree starting with the heap's min element.
        while nodes_heap:
            mindist, idx, typ = heapq.heappop(nodes_heap)
            if typ == INT_NODE:
                node = self.nodes[idx]
                for child in self.children(idx):
                    prep = self.preps[child]
                    heapq.heappush(nodes_heap, (refprep.mindist(prep), child,
                                                int(self.nodes[child].isleaf)))
            elif typ == LEAF_NODE:  # Leaf node. Push the leaves on heap.
                node = self.nodes[idx]
                for idx, prep in self.leaf(node.child):
                    heapq.heappush(nodes_heap,
                                   (refprep.mindist(prep), idx, LEAF_OBJ))
            elif typ == LEAF_OBJ:  # RepGeom. Push shape on heap.
                yield (idx, mindist)
            else:
                raise IndexError("Node type not recognized")


# =========================  Divisive KMeans  =================================

# Divisive KMeans builds an IndexTree based on divisive hierarchical clustering
# via Kmeans clustering at each level.


class DKMeans(IndexTree):
    """Index based on Divisive Hierarchical Clustering via KMeans."""
    
    # Node extends BaseNode
    Node = collections.namedtuple("Node", ' '.join(BaseNode._fields)
                                          + ' centroid_x centroid_y')

    @staticmethod
    # KMeans clustering algorithm, here via scikit-learn
    def _cluster(rgeoms, n_clusters=16, n_init=3, n_jobs=1):
        cluster_model = sklearn.cluster.KMeans(
            n_clusters=n_clusters,
            random_state=0,
            n_init=n_init,
            n_jobs=n_jobs,
        )
        centers = [g.center() for _, g in rgeoms.items()]
        t1 = time.perf_counter()
        fitted = cluster_model.fit(centers)
        t2 = time.perf_counter()
        clust = {k: v for k, v in zip(rgeoms.keys(), fitted.labels_)}
        return (clust, fitted.cluster_centers_)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stats = {
            "kmeans_time": 0.,
            "mean_relative_overlap": 0.,
            "number_nodes_visited": 0,
            "number_overflows": 0,
            "max_depth": 0,
            "number_nodes": 0,
            "number_leaves": 0,
        }

    def bulk_update(self, pgeoms, n_jobs=1):
        '''Create the whole index based on `geoms`.'''
        if len(pgeoms) == 0:
            return
        if not self.isempty:
            raise NotImplementedError("Updates on non-empty Index Tree are "
                                      "not yet implemented")
        self._enclose_class = type(pgeoms[0])
        root = self.Node(0, 0, 0, 0., 0.)
        self.insert_node(root, self._enclose_class.merge(pgeoms.values()))
        self._build(pgeoms, 0, n_jobs)

    def _build(self, pgeoms, idx, n_jobs=1):
        '''Builds rectangles `pgeoms` into node indexed `idx`.'''
        if len(pgeoms) <= self.max_leaves:
            # Update node so that it is pointing to leaf.
            self.nodes[idx] = self.nodes[idx]._replace(
                child=self.leaf_count, isleaf=1)
            self.insert_leaf(pgeoms)
        else:
            clust, centers = self._cluster(pgeoms,
                                           n_clusters=self.max_children)
            # Groupby the geometries by cluster.
            # We ensure the order of groups by casting to OrderedDict
            groups = collections.OrderedDict(
                toolz.valmap(collections.OrderedDict, 
                             toolz.groupby(lambda x: clust[x[0]],
                                           pgeoms.items())
                             )
            )
            # Parallel or not
            if n_jobs == 1:
                # Create children and update parent node.
                self.nodes[idx] = self.nodes[idx]._replace(
                    child=self.node_count,)
                old_count = self.node_count
                for e, (i, grp) in enumerate(groups.items()):
                    if e + 1 >= self.max_children:  # last child
                        new_node = self.Node(0, 0, 0, *centers[i])
                    else:
                        new_node = self.Node(0, self.node_count+1, 0,
                                             *centers[i])
                    self.insert_node(new_node,
                                     self._enclose_class.merge(grp.values())
                                     )
                for e, grp in enumerate(groups.values()):
                    self._build(grp, old_count + e, n_jobs=1)
            else:
                # Bugs in the parallel version, probably in merging trees.
                import multiprocessing
                import dill as pickle
                with multiprocessing.Pool(n_jobs) as pool:
                    child_trees = pool.map(self._index_task,
                                           [g for _, g in groups])
                self.merge(child_trees)

    def _index_task(self, rgeoms):
        '''Builds a sub-ClustTree for parallel computation.'''
        tree = ClustTree(self.rcls, self.max_children, self.max_leaves,
                         self.n_init, 1)
        tree.index(rgeoms)
        return tree

    def _merge(self, trees):
        '''Merge trees under a root node.'''
        for tree in trees:
            # increment indices for nodes and leaves.
            for idx in range(tree.node_count):
                node = tree.nodes.read(idx)
                if node.isleaf:
                    self.nodes.extend(
                        [node.child + self.leaf_count, node.sibling + self.node_count, node.leaf])
                else:
                    self.nodes.extend([node.child + self.node_count, node.sibling + self.node_count, node.leaf])
            self.node_count += tree.node_count    
            self.leaf_count += tree.leaf_count
            self.reps.extend(tree.reps.pool)
            self.centroids.extend(tree.centroids.pool)
