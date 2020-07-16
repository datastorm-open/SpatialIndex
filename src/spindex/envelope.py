import numpy


class EnvelopeVect:
    pass


class AAMBRVect(EnvelopeVect):
    def __init__(self, mins, maxs):
        # Assert ndims and len
        self.mins = mins
        self.maxs = maxs

    def __getitem__(self, idx):
        return self.__class__(self.mins[idx, :], self.maxs[idx, :])

    def __len__(self):
        return self.mins.shape[0]

    @property
    def ndims(self):
        return self.mins.shape[1]

    def mergeby(self, indexes):
        mask = numpy.repeat(indexes.mask, self.ndims).reshape(*indexes.shape, self.ndims)
        return self.__class__(
            mins=numpy.ma.array(self.mins[indexes], mask=mask).min(axis=1).data,
            maxs=numpy.ma.array(self.maxs[indexes], mask=mask).max(axis=1).data,
        )
        # return self.__class__(
        #     mins=numpy.array([self.mins[idx, :].min(axis=0) for idx in indexes]),
        #     maxs=numpy.array([self.maxs[idx, :].max(axis=0) for idx in indexes]),
        # )

    @property
    def centers(self):
        return 0.5 * (self.mins + self.maxs)

    def check_dims(self, other):
        if self.ndims != other.ndims:
            raise ValueError(
                "Incompatible number of dimensions {} and {} in {}.intersects."
                .format(self.ndims, other.ndims, self.__class__)
            )

    def _dist_by_dims(self, other):
        self_shape = (len(self), 1, self.ndims)
        other_shape = (1, len(other), other.ndims)
        dist = numpy.concatenate([
            numpy.abs(self.mins.reshape(self_shape)
                      - other.maxs.reshape(other_shape))[numpy.newaxis, ...],
            numpy.abs(self.maxs.reshape(self_shape)
                      - other.mins.reshape(other_shape))[numpy.newaxis, ...],
        ])
        return dist

    def distance(self, other):
        self.check_dims(other)
        dist = self._dist_by_dims(other).min(axis=0)
        dist[self._intersects_by_dims(other)] = 0.
        return numpy.sqrt((dist**2).sum(axis=2))

    def maxmindist(self, other):
        self.check_dims(other)
        dist = self._dist_by_dims(other).max(axis=0)
        return numpy.sqrt((dist**2).sum(axis=2))

    def bound_dist(self, other):
        """
        Perform both distance and maxmindist in a slightly more efficient way.
        """
        self.check_dims(other)
        dist = self._dist_by_dims(other)
        mindist = dist.min(axis=0)
        mindist[self._intersects_by_dims(other)] = 0.
        return (numpy.sqrt((mindist**2).sum(axis=2)),
                numpy.sqrt((dist.max(axis=0)**2).sum(axis=2)), )

    def _intersects_by_dims(self, other):
        # Uses broadcasting to vectorize comparisons between all pairs of self
        # and other
        self_shape = (len(self), 1, self.ndims)
        other_shape = (1, len(other), other.ndims)
        return (
            (self.mins.reshape(self_shape) < other.maxs.reshape(other_shape))
            & (self.maxs.reshape(self_shape) > other.mins.reshape(other_shape))
        )

    def intersects(self, other):
        self.check_dims(other)
        return self._intersects_by_dims(other).all(axis=2)
