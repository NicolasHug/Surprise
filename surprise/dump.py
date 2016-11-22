"""
The :mod:`dump` module defines the :func:`dump` function.
"""

import pickle


def dump(file_name, predictions, trainset=None, algo=None):
    """Dump a list of :obj:`predictions
    <surprise.prediction_algorithms.predictions.Prediction>` for future
    analysis, using Pickle.

    If needed, the :class:`trainset <surprise.dataset.Trainset>` object and the
    algorithm can also be dumped. What is dumped is a dictionnary with keys
    ``'predictions``, ``'trainset'``, and ``'algo'``.

    The dumped algorithm won't be a proper :class:`algorithm
    <surprise.prediction_algorithms.algo_base.AlgoBase>` object but simply a
    dictionnary with the algorithm attributes as keys-values (technically, the
    ``algo.__dict__`` attribute).

    See :ref:`User Guide <dumping>` for usage.

    Args:
        file_name(str): The name (with full path) specifying where to dump the
            predictions.

        predictions(list of :obj:`Prediction\
            <surprise.prediction_algorithms.predictions.Prediction>`): The
            predictions to dump.
        trainset(:class:`Trainset <surprise.dataset.Trainset>`, optional): The
            trainset to dump.
        algo(:class:`Algorithm\
            <surprise.prediction_algorithms.algo_base.AlgoBase>`, optional):
            algorithm to dump.
    """

    dump_obj = dict()

    dump_obj['predictions'] = predictions

    if trainset is not None:
        dump_obj['trainset'] = trainset

    if algo is not None:
        dump_obj['algo'] = algo.__dict__  # add algo attributes
        dump_obj['algo']['name'] = algo.__class__.__name__

    pickle.dump(dump_obj, open(file_name, 'wb'))
    print('The dump has been saved as file', file_name)
