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
        return self.__class__(
            mins=numpy.array([self.mins[idx, :].min(axis=0) for idx in indexes]),
            maxs=numpy.array([self.maxs[idx, :].max(axis=0) for idx in indexes]),
        )

    @property
    def centers(self):
        return 0.5 * (self.mins + self.maxs)

    def intersects(self, other):
        if self.ndims != other.ndims:
            raise ValueError(
                "Incompatible number of dimensions {} and {} in {}.intersects."
                .format(self.ndims, other.ndims, self.__class__)
            )
        xshape = (len(self), 1, self.ndims)
        yshape = (1, len(other), other.ndims)
        return (
            (self.mins.reshape(xshape) < other.maxs.reshape(yshape))
            & (self.maxs.reshape(xshape) > other.mins.reshape(yshape))
        ).all(axis=2)
