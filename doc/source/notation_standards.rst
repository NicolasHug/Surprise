.. _notation_standards:

Notation standards
==================

In the documentation, you will find the following notation:

* :math:`R` : the set of all ratings.
* :math:`R_{train}`, :math:`R_{test}` and :math:`\hat{R}` denote the training
  set, the test set, and the set of predicted ratings.
* :math:`U` : the set of all users. :math:`u` and :math:`v` denotes users.
* :math:`I` : the set of all items. :math:`i` and :math:`j` denotes items.
* :math:`U_i` : the set of all users that have rated item :math:`i`.
* :math:`U_{ij}` : the set of all users that have rated both items :math:`i`
  and :math:`j`.
* :math:`I_u` : the set of all items rated by user :math:`u`.
* :math:`I_{uv}` : the set of all items rated by both users :math:`u`
  and :math:`v`.
* :math:`r_{ui}` : the *true* rating of user :math:`u` for item
  :math:`i`.
* :math:`\hat{r}_{ui}` : the *estimated* rating of user :math:`u` for item
  :math:`i`.
* :math:`b_{ui}` : the baseline rating of user :math:`u` for item :math:`i`.
* :math:`\mu` : the mean of all ratings.
* :math:`\mu_u` : the mean of all ratings given by user :math:`u`.
* :math:`\mu_i` : the mean of all ratings given to item :math:`i`.
* :math:`N_i^k(u)` : the :math:`k` nearest neighbors of user :math:`u` that
  have rated item :math:`i`. This set is computed using a :mod:`similarity
  metric <pyrec.similarities>`.
* :math:`N_u^k(i)` : the :math:`k` nearest neighbors of item :math:`i` that
  are rated by user :math:`u`. This set is computed using a :py:mod:`similarity
  metric <pyrec.similarities>`.

**Important note**:

A lot of prediction algorithms are symetric: they can be based on users or on
items. For example, a basic *k*-NN algorithm can predict either:

.. math::
  \hat{r}_{ui} = \frac{
  \sum\limits_{v \in N^k_i(u)} \text{sim}(u, v) \cdot r_{vi}}
  {\sum\limits_{v \in N^k_i(u)} \text{sim}(u, v)}

or

.. math::
  \hat{r}_{ui} = \frac{
  \sum\limits_{j \in N^k_u(i)} \text{sim}(i, j) \cdot r_{uj}}
  {\sum\limits_{j \in N^k_u(i)} \text{sim}(i, j)}

depending on wether the similarities are computed between users or between
items. To unify both notations and avoid writing the same code multiple times,
we factorized this into a single notation that is context dependent:

* :math:`x` denotes the entity on which similarities are computed, be it users
  or items ;
* :math:`y` denotes the other entity.

Both formulae above now simply become:

.. math::
  \hat{r}_{xy} = \frac{
  \sum\limits_{x' \in N^k_y(x)} \text{sim}(x, x') \cdot r_{x'y}}
  {\sum\limits_{x' \in N^k_y(x)} \text{sim}(x, x')}.

Likewise, :math:`Y_{xx'}` may denote either :math:`U_{ij}` or :math:`I_{uv}`
depending on the context.

