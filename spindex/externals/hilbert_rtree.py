import numpy

class HilbertCurve:
    @staticmethod
    def _binary_repr(num, width):
        """Return a binary string representation of `num` zero padded to `width`
        bits."""
        return format(num, 'b').zfill(width)

    def __init__(self, p, n, min_geo, max_geo):
        """Initialize a hilbert curve with,
        Args:
            p (int): iterations to use in the hilbert curve
            n (int): number of dimensions
        """
        self.depth = p
        self.dim = n
        self.min_geo = min_geo
        self.max_geo = max_geo

    def _coordinates_to_geospatial(self, x):
        return [((2**self.depth - 1 - x[i]) * self.min_geo[i]
                 + x[i] * self.max_geo[i]) / (2**self.depth - 1)
                for i in range(self.dim)]

    def _geospatial_to_coordinates(self, g):
        return [(g[i] - self.min_geo[i]) * (2**self.depth - 1)
                / (self.max_geo[i] - self.min_geo[i])
                for i in range(self.dim)]

    def _hilbert_integer_to_transpose(self, h):
        """Store a hilbert integer (`h`) as its transpose (`x`).
        Args:
            h (int): integer distance along hilbert curve
        Returns:
            x (list): transpose of h
                      (n components with values between 0 and 2**p-1)
        """
        h_bit_str = self._binary_repr(h, self.depth*self.dim)
        x = [int(h_bit_str[i::self.dim], 2) for i in range(self.dim)]
        return x

    def _transpose_to_hilbert_integer(self, x):
        """Restore a hilbert integer (`h`) from its transpose (`x`).
        Args:
            x (list): transpose of h
                      (n components with values between 0 and 2**p-1)
        Returns:
            h (int): integer distance along hilbert curve
        """
        x_bit_str = [self._binary_repr(x[i], self.depth) for i in range(self.dim)]
        h = int(''.join([y[i] for i in range(self.depth) for y in x_bit_str]), 2)
        return h

    def decode(self, h):
        """Return the coordinates for a given hilbert distance.
        Args:
            h (int): integer distance along hilbert curve
        Returns:
            x (list): transpose of h
                      (n components with values between 0 and 2**p-1)
        """
        max_h = 2**(self.depth * self.dim) - 1
        if h > max_h:
            raise ValueError('h={} is greater than 2**(p*N)-1={}'.format(h, max_h))

        x = self._hilbert_integer_to_transpose(h)
        Z = 2 << (self.depth-1)

        # Gray decode by H ^ (H/2)
        t = x[self.dim-1] >> 1
        for i in range(self.dim-1, 0, -1):
            x[i] ^= x[i-1]
        x[0] ^= t

        # Undo excess work
        Q = 2
        while Q != Z:
            P = Q - 1
            for i in range(self.dim-1, -1, -1):
                if x[i] & Q:
                    # invert
                    x[0] ^= P
                else:
                    # exchange
                    t = (x[0] ^ x[i]) & P
                    x[0] ^= t
                    x[i] ^= t
            Q <<= 1

        # done
        return self._coordinates_to_geospatial(x)

    def encode(self, g):
        """Return the hilbert distance for a given set of coordinates.
        Args:
            x (list): transpose of h
                      (n components with values between 0 and 2**p-1)
        Returns:
            h (int): integer distance along hilbert curve
        """
        x = [int(c) for c in self._geospatial_to_coordinates(g)]
        if len(x) != self.dim:
            raise ValueError('x={} must have N={} dimensions'.format(x, self.dim))

        max_x = 2**self.depth - 1
        if any(elx > max_x for elx in x):
            raise ValueError(
                'invalid coordinate input x={}.  one or more dimensions have a '
                'value greater than 2**p-1={}'.format(x, max_x))

        M = 1 << (self.depth - 1)

        # Inverse undo excess work
        Q = M
        while Q > 1:
            P = Q - 1
            for i in range(self.dim):
                if x[i] & Q:
                    x[0] ^= P
                else:
                    t = (x[0] ^ x[i]) & P
                    x[0] ^= t
                    x[i] ^= t
            Q >>= 1

        # Gray encode
        for i in range(1, self.dim):
            x[i] ^= x[i-1]
        t = 0
        Q = M
        while Q > 1:
            if x[self.dim-1] & Q:
                t ^= Q - 1
            Q >>= 1
        for i in range(self.dim):
            x[i] ^= t

        h = self._transpose_to_hilbert_integer(x)
        return h


# Node = idx, pointers to first_child and to next_sibling
# Node_pool at idx will contain these 2 indices.
# Node id = 0 means empty (sentinel)
class HRTree():
    def __init__(self, bbox=(97000, 6050000, 1250000, 7115000)):
        self.max_children = 16
        self.min_children = 12
        self.max_leaf = 64
        self.count = 0
        self.leaf_count = 0
        self.rect_pool = numpy.empty((0, 4), dtype=float)    
        self.node_pool = numpy.zeros((1, 2), dtype='uint')   
        self.hcode_pool = numpy.array(dtype='uint')  
        self.leaf_pool = []
        self.hilbert = HilbertCurve(24, 2, bbox[:2], bbox[2:])
        self.stats = {
            "overflow_f": 0,
            "avg_overflow_t_f": 0.0,
            "longest_overflow": 0.0,
            "longest_kmeans": 0.0,
            "sum_kmeans_iter_f": 0,
            "count_kmeans_iter_f": 0,
            "avg_kmeans_iter_f": 0.0
        }

    def first_child(self, node):
        yield self.node_pool[2*(node-1)]

    def is_leaf(self, node):
        return numpy.isnan(self.first_child(node))

    def next_sibling(self, node):
        yield self.node_pool[2*(node-1) + 1]

    def children(self, node):
        child = self.first_child(node)
        while not numpy.isnan(child):
            yield child
            child = self.next_sibling(child)

    def build(self, geometry):
        rcenters = geometry.apply(lambda g: (sum(g.bounds[::2])/2,
                                             sum(g.bounds[1::2])/2)
                                  )
        hcodes = rcenters.apply(self.hilbert.encode).sort_values()
        words = pandas.Series(0, index=geometry.index)

        # Each level of the tree is clustered recursively.
        depth = int(len(geometry)/self.max_leaf)
        for i in range(depth):
            words = words + 2*(depth-i)*words.groupby(words).apply(
                lambda s: mod_kmeans(hcodes.loc[s.index], self.max_children)
            )
        
        # for i in hcodes.index:
        #     self.insert(i, geometry.loc[i].bounds, hcodes.loc[i])

    def insert(self, idx, rect, hcode):
        leaf = self.choose_leaf(hcode)
        s

def breath_first_iterator(tree, root):
    '''Breath-first iterator of tree.'''
    fifo = [root]
    while fifo:
        node = fifo.pop(0)
        yield node
        if not tree.is_leaf(node):
            fifo.extend(tree.children(node))


def depth_first_iterator(tree, root):
    '''Depth-first iterator of tree.'''
    filo = [root]
    while filo:
        node = filo.pop()
        yield node
        if not tree.is_leaf(node):
            filo.extend(tree.children(node))


