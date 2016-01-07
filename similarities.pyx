"""
The :mod:`similarities` module includes tools to compute similarity metrics
between users or items. Please refer to the :ref:`notation standards
<notation_standards>`.
"""

cimport numpy as np
import numpy as np
from itertools import combinations

def cosine(nX, yr):
    """Compute the cosine similarity between all pairs of xs.

    Only *common* users (or items) are taken into account:

    :math:`\\text{cos_sim}(x, x') = \\frac{
    \\sum\\limits_{y \in Y_{xx'}} r_{xy} \cdot r_{x'y}}
    {\\sqrt{\\sum\\limits_{y \in Y_{xx'}} r_{xy}^2} \cdot
    \\sqrt{\\sum\\limits_{y \in Y_{xx'}} r_{x'y}^2}
    }`

    See details on `Wikipedia
    <https://en.wikipedia.org/wiki/Cosine_similarity#Definition>`_.
    """

    # sum (r_xy * r_x'y) for common ys
    cdef np.ndarray[np.int_t, ndim = 2] prods     = np.zeros((nX, nX), np.int)
    # number of common ys
    cdef np.ndarray[np.int_t, ndim = 2] freq      = np.zeros((nX, nX), np.int)
    # sum (r_xy ^ 2) for common ys
    cdef np.ndarray[np.int_t, ndim = 2] sqi       = np.zeros((nX, nX), np.int)
    # sum (r_x'y ^ 2) for common ys
    cdef np.ndarray[np.int_t, ndim = 2] sqj       = np.zeros((nX, nX), np.int)
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim = 2] simMat = np.zeros((nX, nX))

    # these variables need to be cdef'd so that array lookup can be fast
    cdef int xi = 0
    cdef int xj = 0
    cdef int r1 = 0
    cdef int r2 = 0

    for y, yRatings in yr.items():
        # combinations are emitted in lexicographic sort order. It's important
        # for the next loop
        for (xi, r1), (xj, r2) in combinations(yRatings, 2):
            freq[xi, xj] += 1
            prods[xi, xj] += r1 * r2
            sqi[xi, xj] += r1**2
            sqj[xi, xj] += r2**2

    for xi in range(nX):
        simMat[xi, xi] = 1
        for xj in range(xi + 1, nX):
            if freq[xi, xj] == 0:
                simMat[xi, xj] = 0
            else:
                denum = np.sqrt(sqi[xi, xj] * sqj[xi, xj])
                simMat[xi, xj] = prods[xi, xj] / denum

            simMat[xj, xi] = simMat[xi, xj]

    return simMat

def msd(nX, yr):
    """Compute the mean squared difference similarity between all pairs of
    xs.

    Only *common* users (or items) are taken into account:

    :math:`\\text{msd}(x, x') = \\frac{1}{|Y_{xx'}|} \cdot
    \\sum\\limits_{y \in Y_{xx'}} (r_{xy} - r_{x'y})^2`

    :math:`\\text{msd_sim}(x, x') = \\left\\{
    \\begin{array}{ll}
    \\frac{1}{\\text{msd}(x, x')} & \mbox{if }\\text{msd}(x, x') \\neq 0 \\\\
    |Y_{xx'}|& \mbox{else (which is quite arbitrary).}
    \end{array}
    \\right.`

    For details, see third definition on `Wikipedia
    <https://en.wikipedia.org/wiki/Root-mean-square_deviation#Formula>`_.

    """

    # sum (r_xy - r_x'y)**2 for common ys
    cdef np.ndarray[np.double_t, ndim = 2] sqDiff = np.zeros((nX, nX), np.double)
    # number of common ys
    cdef np.ndarray[np.int_t,    ndim = 2] freq   = np.zeros((nX, nX), np.int)
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim = 2] simMat = np.zeros((nX, nX))

    # these variables need to be cdef'd so that array lookup can be fast
    cdef int xi = 0
    cdef int xj = 0
    cdef int r1 = 0
    cdef int r2 = 0

    for y, yRatings in yr.items():
        for (xi, r1), (xj, r2) in combinations(yRatings, 2):
            sqDiff[xi, xj] += (r1 - r2)**2
            freq[xi, xj] += 1

    for xi in range(nX):
        simMat[xi, xi] = 100 # completely arbitrary and useless anyway
        for xj in range(xi, nX):
            if sqDiff[xi, xj] == 0: # return number of common ys (arbitrary)
                simMat[xi, xj] = freq[xi, xj]
            else:  # return inverse of MSD
                simMat[xi, xj] = freq[xi, xj] / sqDiff[xi, xj]

            simMat[xj, xi] = simMat[xi, xj]

    return simMat

def msdClone(nX, xr, rm):
    """compute the 'clone' mean squared difference similarity between all
    pairs of xs. Some properties as for MSDSim apply. Not optimal at all"""

    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim = 2] simMat = np.zeros((nX, nX))

    cdef int xi = 0
    cdef int xj = 0
    cdef int y = 0


    for xi in range(nX):
        simMat[xi, xi] = 100 # completely arbitrary and useless anyway
        for xj in range(xi, nX):
            # comon ys for xi and xj
            Yij = [y for (y, _) in xr[xi] if rm[xj, y] > 0]

            if not Yij:
                simMat[xi, xj] = 0
                continue

            meanDiff = np.mean([rm[xi, y] - rm[xj, y] for y in Yij])
            # sum of squared differences:
            ssd = sum((rm[xi, y] - rm[xj, y] - meanDiff)**2 for y in Yij)
            if ssd == 0:
                simMat[xi, xj] = len(Yij) # well... ok.
            else:
                simMat[xi, xj] = len(Yij) / ssd

            simMat[xj, xi] = simMat[xi, xj]

    return simMat

def pearson(nX, yr):
    """compute the pearson corr coeff between all pairs of xs.

    Only *common* users (or items) are taken into account:

    :math:`\\text{pearson_sim}(x, x') = \\frac{
    \\sum\\limits_{y \in Y_{xx'}} (r_{xy} -  \mu_x) \cdot (r_{x'y} - \mu_{x'})}
    {\\sqrt{\\sum\\limits_{y \in Y_{xx'}} (r_{xy} -  \mu_x)^2} \cdot
    \\sqrt{\\sum\\limits_{y \in Y_{xx'}} (r_{x'y} -  \mu_{x'})^2}
    }`

    See details on `Wikipedia
    <https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#For_a_sample>`_.
    """

    # number of common ys
    cdef np.ndarray[np.int_t,    ndim = 2] freq   = np.zeros((nX, nX), np.int)
    # sum (r_xy * r_x'y) for common ys
    cdef np.ndarray[np.int_t,    ndim = 2] prods = np.zeros((nX, nX), np.int)
    # sum (rxy ^ 2) for common ys
    cdef np.ndarray[np.int_t,    ndim = 2] sqi = np.zeros((nX, nX), np.int)
    # sum (rx'y ^ 2) for common ys
    cdef np.ndarray[np.int_t,    ndim = 2] sqj = np.zeros((nX, nX), np.int)
    # sum (rxy) for common ys
    cdef np.ndarray[np.int_t,    ndim = 2] si = np.zeros((nX, nX), np.int)
    # sum (rx'y) for common ys
    cdef np.ndarray[np.int_t,    ndim = 2] sj = np.zeros((nX, nX), np.int)
    # the similarity matrix
    cdef np.ndarray[np.double_t, ndim = 2] simMat = np.zeros((nX, nX))

    # these variables need to be cdef'd so that array lookup can be fast
    cdef int xi = 0
    cdef int xj = 0
    cdef int r1 = 0
    cdef int r2 = 0

    for y, yRatings in yr.items():
        for (xi, r1), (xj, r2) in combinations(yRatings, 2):
            # note : accessing and updating elements takes a looooot of
            # time. Yet defaultdict is still faster than a numpy array...
            prods[xi, xj] += r1 * r2
            freq[xi, xj] += 1
            sqi[xi, xj] += r1**2
            sqj[xi, xj] += r2**2
            si[xi, xj] += r1
            sj[xi, xj] += r2

    for xi in range(nX):
        simMat[xi, xi] = 1
        for xj in range(xi + 1, nX):
            n = freq[xi, xj]
            num = n * prods[xi, xj] - si[xi, xj] * sj[xi, xj]
            denum = np.sqrt((n * sqi[xi, xj] - si[xi, xj]**2) *
                            (n * sqj[xi, xj] - sj[xi, xj]**2))
            if denum == 0:
                simMat[xi, xj] = 0
            else:
                simMat[xi, xj] = num / denum

            simMat[xj, xi] = simMat[xi, xj]

    return simMat
