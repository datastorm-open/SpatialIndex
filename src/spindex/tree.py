import collections

import numpy


class RTree():
    """
    Highly efficient R-Tree structure using arrays.

    Arrays are used instead of the classically doubly linked list of siblings at a tree levels.

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
    def depth(self):
        return len(self.envelopes)

    @property
    def is_empty(self):
        return self.depth == 0

    def __len__(self):
        """Returns the number of leaves."""
        return len(self.envelopes[-1])

    def query(self, obj, op, sparse=False):
        """
        Bulk search query with given relationship.

        Parameters:
            obj (EnvelopeVect): query objects
            predicate (str): One of {}
        """.format(self.valid_predicates)
        empty_result = numpy.array([], dtype='int')
        if self.is_empty:
            return [empty_result for _ in range(len(obj))]
        if op not in self.valid_predicates:
            raise ValueError(
                "Invalid predicate {}: must be one of {}"
                .format(op, self.valid_predicates)
            )

        # Cascading through tree levels to get matching indexes.
        # Path corresponds to a search path consisting of indexes in both self
        #   and obj.
        Path = collections.namedtuple("QueryResult", "this that")
        paths = [Path(this=numpy.arange(len(self.envelopes[0])),
                      that=numpy.arange(len(obj)))]
        for envelopes, children in zip(self.envelopes, self.children):
            next_paths = []
            for path in paths:
                pred = getattr(obj[path.that], op)(envelopes[path.this])
                # Numpy style groupby pred's rows -> list of index arrays
                hashes = numpy.apply_along_axis(lambda a: a.tobytes(), 1, pred)
                arg = numpy.argsort(hashes)
                _, ind = numpy.unique(hashes[arg], return_index=True)
                groups = numpy.split(arg, ind[1:])
                # Each group corresponds to a search path
                next_paths.extend([
                    Path(this=children[path.this, :][pred[g[0], :]]
                         .compressed(),
                         that=path.that[g])
                    for g in groups
                ])
                # next_paths.extend([
                #     Path(this=numpy.concatenate(
                #             [child for child, p in zip(children, pred[g[0], :])
                #              if p]),
                #          that=path.that[g])
                #     for g in groups
                # ])
            paths = next_paths
        if sparse:
            return paths
        # Resulting valid paths are reordered in original obj order.
        result = [empty_result]*len(obj)
        for path in paths:
            for i in path.that:
                result[i] = path.this
        return result
