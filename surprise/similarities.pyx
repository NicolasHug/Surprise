"""
The :mod:`similarities <surprise.similarities>` module includes tools to
compute similarity metrics between users or items. You may need to refer to the
:ref:`notation_standards` page. See also the
:ref:`similarity_measures_configuration` section of the User Guide.

Available similarity measures:

.. autosummary::
    :nosignatures:

    cosine
    msd
    pearson
    pearson_baseline
"""


cimport numpy as np  # noqa
import numpy as np
from libc.math cimport sqrt


def cosine(int n_x, yr, int min_support):
    """Compute the cosine similarity between all pairs of users (or items).

    Only **common** users (or items) are taken into account. The cosine
    similarity is defined as:

    .. math::
        \\text{cosine_sim}(u, v) = \\frac{
        \\sum\\limits_{i \\in I_{uv}} r_{ui} \\cdot r_{vi}}
        {\\sqrt{\\sum\\limits_{i \\in I_{uv}} r_{ui}^2} \\cdot
        \\sqrt{\\sum\\limits_{i \\in I_{uv}} r_{vi}^2}
        }

    or

    .. math::
        \\text{cosine_sim}(i, j) = \\frac{
        \\sum\\limits_{u \\in U_{ij}} r_{ui} \\cdot r_{uj}}
        {\\sqrt{\\sum\\limits_{u \\in U_{ij}} r_{ui}^2} \\cdot
        \\sqrt{\\sum\\limits_{u \\in U_{ij}} r_{uj}^2}
        }

    depending on the ``user_based`` field of ``sim_options`` (see
    :ref:`similarity_measures_configuration`).

    For details on cosine similarity, see on `Wikipedia
    <https://en.wikipedia.org/wiki/Cosine_similarity#Definition>`__.
    """

    # sum (r_xy * r_x'y) for common ys
    cdef double [:, ::1] prods = np.zeros((n_x, n_x), np.double)
    # number of common ys
    cdef long [:, ::1] freq = np.zeros((n_x, n_x), np.int_)
    # sum (r_xy ^ 2) for common ys
    cdef double [:, ::1] sqi = np.zeros((n_x, n_x), np.double)
    # sum (r_x'y ^ 2) for common ys
    cdef double [:, ::1] sqj = np.zeros((n_x, n_x), np.double)
    # the similarity matrix
    cdef double [:, ::1] sim = np.zeros((n_x, n_x), np.double)

    cdef int xi, xj, y
    cdef double ri, rj
    cdef int min_sprt = min_support

    for y, y_ratings in yr.items():
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                freq[xi, xj] += 1
                prods[xi, xj] += ri * rj
                sqi[xi, xj] += ri**2
                sqj[xi, xj] += rj**2

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                denum = sqrt(sqi[xi, xj] * sqj[xi, xj])
                sim[xi, xj] = prods[xi, xj] / denum

            sim[xj, xi] = sim[xi, xj]

    return np.asarray(sim)


def msd(int n_x, yr, int min_support):
    """Compute the Mean Squared Difference similarity between all pairs of
    users (or items).

    Only **common** users (or items) are taken into account. The Mean Squared
    Difference is defined as:

    .. math ::
        \\text{msd}(u, v) = \\frac{1}{|I_{uv}|} \\cdot
        \\sum\\limits_{i \\in I_{uv}} (r_{ui} - r_{vi})^2

    or

    .. math ::
        \\text{msd}(i, j) = \\frac{1}{|U_{ij}|} \\cdot
        \\sum\\limits_{u \\in U_{ij}} (r_{ui} - r_{uj})^2

    depending on the ``user_based`` field of ``sim_options`` (see
    :ref:`similarity_measures_configuration`).

    The MSD-similarity is then defined as:

    .. math ::
        \\text{msd_sim}(u, v) &= \\frac{1}{\\text{msd}(u, v) + 1}\\\\
        \\text{msd_sim}(i, j) &= \\frac{1}{\\text{msd}(i, j) + 1}

    The :math:`+ 1` term is just here to avoid dividing by zero.


    For details on MSD, see third definition on `Wikipedia
    <https://en.wikipedia.org/wiki/Root-mean-square_deviation#Formula>`__.

    """

    # sum (r_xy - r_x'y)**2 for common ys
    cdef double [:, ::1] sq_diff = np.zeros((n_x, n_x), np.double)
    # number of common ys
    cdef long [:, ::1] freq = np.zeros((n_x, n_x), np.int_)
    # the similarity matrix
    cdef double [:, ::1] sim = np.zeros((n_x, n_x), np.double)

    cdef int xi, xj
    cdef double ri, rj
    cdef int min_sprt = min_support

    for y, y_ratings in yr.items():
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                sq_diff[xi, xj] += (ri - rj)**2
                freq[xi, xj] += 1

    for xi in range(n_x):
        sim[xi, xi] = 1  # completely arbitrary and useless anyway
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                # return inverse of (msd + 1) (+ 1 to avoid dividing by zero)
                sim[xi, xj] = 1 / (sq_diff[xi, xj] / freq[xi, xj] + 1)

            sim[xj, xi] = sim[xi, xj]

    return np.asarray(sim)


def pearson(int n_x, yr, int min_support):
    """Compute the Pearson correlation coefficient between all pairs of users
    (or items).

    Only **common** users (or items) are taken into account. The Pearson
    correlation coefficient can be seen as a mean-centered cosine similarity,
    and is defined as:

    .. math ::
        \\text{pearson_sim}(u, v) = \\frac{ \\sum\\limits_{i \\in I_{uv}}
        (r_{ui} -  \\mu_u) \\cdot (r_{vi} - \\mu_{v})} {\\sqrt{\\sum\\limits_{i
        \\in I_{uv}} (r_{ui} -  \\mu_u)^2} \\cdot \\sqrt{\\sum\\limits_{i \\in
        I_{uv}} (r_{vi} -  \\mu_{v})^2} }

    or

    .. math ::
        \\text{pearson_sim}(i, j) = \\frac{ \\sum\\limits_{u \\in U_{ij}}
        (r_{ui} -  \\mu_i) \\cdot (r_{uj} - \\mu_{j})} {\\sqrt{\\sum\\limits_{u
        \\in U_{ij}} (r_{ui} -  \\mu_i)^2} \\cdot \\sqrt{\\sum\\limits_{u \\in
        U_{ij}} (r_{uj} -  \\mu_{j})^2} }

    depending on the ``user_based`` field of ``sim_options`` (see
    :ref:`similarity_measures_configuration`).


    Note: if there are no common users or items, similarity will be 0 (and not
    -1).

    For details on Pearson coefficient, see `Wikipedia
    <https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#For_a_sample>`__.

    """
    # number of common ys
    cdef long [:, ::1] freq = np.zeros((n_x, n_x), np.int_)
    # sum (r_xy * r_x'y) for common ys
    cdef double [:, ::1] prods = np.zeros((n_x, n_x), np.double)
    # sum (rxy ^ 2) for common ys
    cdef double [:, ::1] sqi = np.zeros((n_x, n_x), np.double)
    # sum (rx'y ^ 2) for common ys
    cdef double [:, ::1] sqj = np.zeros((n_x, n_x), np.double)
    # sum (rxy) for common ys
    cdef double [:, ::1] si = np.zeros((n_x, n_x), np.double)
    # sum (rx'y) for common ys
    cdef double [:, ::1] sj = np.zeros((n_x, n_x), np.double)
    # the similarity matrix
    cdef double [:, ::1] sim = np.zeros((n_x, n_x), np.double)

    cdef int xi, xj, y, n
    cdef double ri, rj, num, denum
    cdef int min_sprt = min_support

    for y, y_ratings in yr.items():
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                prods[xi, xj] += ri * rj
                freq[xi, xj] += 1
                sqi[xi, xj] += ri**2
                sqj[xi, xj] += rj**2
                si[xi, xj] += ri
                sj[xi, xj] += rj

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):

            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                n = freq[xi, xj]
                num = n * prods[xi, xj] - si[xi, xj] * sj[xi, xj]
                denum = sqrt((n * sqi[xi, xj] - si[xi, xj]**2) *
                             (n * sqj[xi, xj] - sj[xi, xj]**2))
                if denum == 0:
                    sim[xi, xj] = 0
                else:
                    sim[xi, xj] = num / denum

            sim[xj, xi] = sim[xi, xj]

    return np.asarray(sim)


def pearson_baseline(
    int n_x,
    yr,
    int min_support,
    double global_mean,
    double [::1] x_biases,
    double [::1] y_biases,
    double shrinkage=100,
):
    """Compute the (shrunk) Pearson correlation coefficient between all pairs
    of users (or items) using baselines for centering instead of means.

    The shrinkage parameter helps to avoid overfitting when only few ratings
    are available (see :ref:`similarity_measures_configuration`).

    The Pearson-baseline correlation coefficient is defined as:

    .. math::
        \\text{pearson_baseline_sim}(u, v) = \\hat{\\rho}_{uv} = \\frac{
            \\sum\\limits_{i \\in I_{uv}} (r_{ui} -  b_{ui}) \\cdot (r_{vi} -
            b_{vi})} {\\sqrt{\\sum\\limits_{i \\in I_{uv}} (r_{ui} -  b_{ui})^2}
            \\cdot \\sqrt{\\sum\\limits_{i \\in I_{uv}} (r_{vi} -  b_{vi})^2}}

    or

    .. math::
        \\text{pearson_baseline_sim}(i, j) = \\hat{\\rho}_{ij} = \\frac{
            \\sum\\limits_{u \\in U_{ij}} (r_{ui} -  b_{ui}) \\cdot (r_{uj} -
            b_{uj})} {\\sqrt{\\sum\\limits_{u \\in U_{ij}} (r_{ui} -  b_{ui})^2}
            \\cdot \\sqrt{\\sum\\limits_{u \\in U_{ij}} (r_{uj} -  b_{uj})^2}}

    The shrunk Pearson-baseline correlation coefficient is then defined as:

    .. math::
        \\text{pearson_baseline_shrunk_sim}(u, v) &= \\frac{|I_{uv}| - 1}
        {|I_{uv}| - 1 + \\text{shrinkage}} \\cdot \\hat{\\rho}_{uv}

        \\text{pearson_baseline_shrunk_sim}(i, j) &= \\frac{|U_{ij}| - 1}
        {|U_{ij}| - 1 + \\text{shrinkage}} \\cdot \\hat{\\rho}_{ij}


    Obviously, a shrinkage parameter of 0 amounts to no shrinkage at all.

    Note: here again, if there are no common users/items, similarity will be 0
    (and not -1).

    Motivations for such a similarity measure can be found on the *Recommender
    System Handbook*, section 5.4.1.
    """

    # number of common ys
    cdef long [:, ::1] freq = np.zeros((n_x, n_x), np.int_)
    # sum (r_xy - b_xy) * (r_x'y - b_x'y) for common ys
    cdef double [:, ::1] prods = np.zeros((n_x, n_x), np.double)
    # sum (r_xy - b_xy)**2 for common ys
    cdef double [:, ::1] sq_diff_i = np.zeros((n_x, n_x), np.double)
    # sum (r_x'y - b_x'y)**2 for common ys
    cdef double [:, ::1] sq_diff_j = np.zeros((n_x, n_x), np.double)
    # the similarity matrix
    cdef double [:, ::1] sim = np.zeros((n_x, n_x), np.double)

    cdef int y, xi, xj
    cdef double ri, rj, diff_i, diff_j, partial_bias
    cdef int min_sprt = min_support
    cdef double global_mean_ = global_mean

    # Need this because of shrinkage. When pearson coeff is zero when support
    # is 1, so that's OK.
    min_sprt = max(2, min_sprt)

    for y, y_ratings in yr.items():
        partial_bias = global_mean_ + y_biases[y]
        for xi, ri in y_ratings:
            for xj, rj in y_ratings:
                freq[xi, xj] += 1
                diff_i = (ri - (partial_bias + x_biases[xi]))
                diff_j = (rj - (partial_bias + x_biases[xj]))
                prods[xi, xj] += diff_i * diff_j
                sq_diff_i[xi, xj] += diff_i**2
                sq_diff_j[xi, xj] += diff_j**2

    for xi in range(n_x):
        sim[xi, xi] = 1
        for xj in range(xi + 1, n_x):
            if freq[xi, xj] < min_sprt:
                sim[xi, xj] = 0
            else:
                sim[xi, xj] = prods[xi, xj] / (sqrt(sq_diff_i[xi, xj] * sq_diff_j[xi, xj]))
                # the shrinkage part
                sim[xi, xj] *= (freq[xi, xj] - 1) / (freq[xi, xj] - 1 + shrinkage)

            sim[xj, xi] = sim[xi, xj]

    return np.asarray(sim)
