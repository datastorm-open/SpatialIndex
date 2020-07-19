import numpy


class EnvelopeVect:
    pass


class AAMBRVect(EnvelopeVect):
    def __init__(self, coords=None, maxs=None, mins=None, interleaved=True):
        """
        Either pass coords and optionally interleaved, or pass it mins and maxs
        separately.
        """
        cases = {
            "mixed": (coords is not None and maxs is None and mins is None), 
            "mins-maxs": (
                maxs is not None and coords is None and mins is not None),
            "coords-maxs": (
                maxs is not None and coords is not None and mins is None),
        }
        if cases["mixed"]:
            self._from_array(coords, interleaved)
        elif cases["coords-maxs"]:
            self._from_mins_maxs(coords, maxs)
        elif cases["mins-maxs"]:
            self._from_mins_maxs(mins, maxs)
        else:
            raise ValueError(
                "Either coords or both mins and maxs must be given (but not "
                "all together)."
            )

    def _from_array(self, coords, interleaved=True):
        coords = numpy.array(coords)
        three_dims = (coords.ndim == 3 and coords.shape[2] == 2)
        two_dims = (coords.ndim == 2 and coords.shape[1] % 2 == 0)
        if not (three_dims or two_dims):
            raise ValueError(
                "Coords third dimension must correspond to mins and "
                "maxs in each coordinate dimension, and must be of "
                "of even length."
            )
        self.coords = coords
        if two_dims: 
            self.coords = coords.reshape(coords.shape[0], -1, 2)
            if not interleaved:  # corners
                self.coords = numpy.swapaxes(self.coords, 1, 2)
        self.ndims = self.coords.shape[1]
        self.mins = self.coords[:,:,0]
        self.maxs = self.coords[:,:,1]

    def _from_mins_maxs(self, mins, maxs):
        if mins.shape != maxs.shape:
            raise ValueError("Mins and maxs must be of same shape")
        self._from_array(
            numpy.concatenate(
                [mins[..., numpy.newaxis], maxs[..., numpy.newaxis]],
                axis=2,
            ),
        )

    def __getitem__(self, idx):
        return self.__class__(self.mins[idx, :], self.maxs[idx, :])

    def __len__(self):
        return self.mins.shape[0]

    def mergeby(self, indexes):
        mask = numpy.repeat(indexes.mask, self.ndims).reshape(*indexes.shape, self.ndims)
        return self.__class__(
            numpy.ma.array(self.mins[indexes], mask=mask).min(axis=1).data,
            numpy.ma.array(self.maxs[indexes], mask=mask).max(axis=1).data,
        )

    @property
    def centers(self):
        return 0.5 * (self.mins + self.maxs)

    def check_dims(self, other):
        if self.ndims != other.ndims:
            raise ValueError(
                "Incompatible number of dimensions {} and {} in {}.intersects."
                .format(self.ndims, other.ndims, self.__class__)
            )

    @property
    def boundary(self):
        """
        Returns:
            Nx2DxDx2 array, where each Dx2 subarray is the AAMBR of a face.
        """
        D = self.ndims
        res = numpy.tile(self.coords, (1, 2*D, 1)).reshape(
                len(self), 2*D,  *self.coords.shape[1:])
        res[:, range(D), range(D), 1] = self.coords[:, :, 0]
        res[:, range(D, 2*D), range(D), 0] = self.coords[:, :, 1]
        return res
        
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

    def maxdist(self, other):
        self.check_dims(other)
        dist = self._dist_by_dims(other).max(axis=0)
        return numpy.sqrt((dist**2).sum(axis=2))

    def maxmindist(self, other):
        left = self.boundary
        right = other.boundary
        left_shape = (
            left.shape[0], 1, left.shape[1], 1, left.shape[2], left.shape[3])
        right_shape = (
            1, right.shape[0], 1, right.shape[1],
            right.shape[2], right.shape[3]
        )
        left = left.reshape(left_shape)
        right = right.reshape(right_shape)
        return (
            numpy.sqrt(
                numpy.max([(left[..., 0] - right[..., 1])**2,
                           (left[..., 1] - right[..., 0])**2], axis=0, )
                .sum(axis=-1)
            )
            .min(axis=(-2, -1))
        )

    def bound_dist(self, other):
        """
        Perform both distance and maxdist in a slightly more efficient way.
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
