.. _notation_standards:

Notation standards, References
==============================

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
* :math:`\sigma_u` : the standard deviation of all ratings given by user :math:`u`.
* :math:`\sigma_i` : the standard deviation of all ratings given to item :math:`i`.
* :math:`N_i^k(u)` : the :math:`k` nearest neighbors of user :math:`u` that
  have rated item :math:`i`. This set is computed using a :mod:`similarity
  metric <surprise.similarities>`.
* :math:`N_u^k(i)` : the :math:`k` nearest neighbors of item :math:`i` that
  are rated by user :math:`u`. This set is computed using a :py:mod:`similarity
  metric <surprise.similarities>`.

.. rubric:: References

Here are the papers used as references in the documentation. Links to pdf files
where added when possible. A simple Google search should lead you easily to the
missing ones :)

.. bibliography:: refs.bib
  :all:
