# courtesy of E. Almamedov with modifications for our purposes

import numpy as np


def expand2d(vect, size2, vertical=True):
    """
    This expands a vector to a 2d-array.

    The result is the same as:

    .. code-block:: python

        if vertical:
            numpy.outer(numpy.ones(size2), vect)
        else:
            numpy.outer(vect, numpy.ones(size2))

    This is a ninja optimization: replace \\*1 with a memcopy, saves 50% of
    time at the ms level.

    :param vect: 1d vector
    :param size2: size of the expanded dimension
    :param vertical: if False the vector is expanded to the first dimension.
        If True, it is expanded to the second dimension.
    """
    size1 = vect.size
    size2 = int(size2)
    if vertical:
        out = np.empty((size2, size1), vect.dtype)
        q = vect.reshape(1, -1)
        q.strides = 0, vect.strides[0]
    else:
        out = np.empty((size1, size2), vect.dtype)
        q = vect.reshape(-1, 1)
        q.strides = vect.strides[0], 0
    out[:, :] = q
    return out
