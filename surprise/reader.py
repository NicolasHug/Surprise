"""This module contains the Reader class."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from .builtin_datasets import BUILTIN_DATASETS


class Reader():
    """The Reader class is used to parse a file containing ratings.

    Such a file is assumed to specify only one rating per line, and each line
    needs to respect the following structure: ::

        user ; item ; rating ; [timestamp]

    where the order of the fields and the separator (here ';') may be
    arbitrarily defined (see below).  brackets indicate that the timestamp
    field is optional.

    For each built-in dataset, Surprise also provides predefined readers which
    are useful if you want to use a custom dataset that has the same format as
    a built-in one (see the ``name`` parameter).


    Args:
        name(:obj:`string`, optional): If specified, a Reader for one of the
            built-in datasets is returned and any other parameter is ignored.
            Accepted values are 'ml-100k', 'ml-1m', and 'jester'. Default
            is ``None``.
        line_format(:obj:`string`): The fields names, in the order at which
            they are encountered on a line. Please note that ``line_format`` is
            always space-separated (use the ``sep`` parameter). Default is
            ``'user item rating'``.
        sep(char): the separator between fields. Example : ``';'``.
        rating_scale(:obj:`tuple`, optional): The rating scale used for every
            rating.  Default is ``(1, 5)``.
        skip_lines(:obj:`int`, optional): Number of lines to skip at the
            beginning of the file. Default is ``0``.

    """

    def __init__(self, name=None, line_format='user item rating', sep=None,
                 rating_scale=(1, 5), skip_lines=0):

        if name:
            try:
                self.__init__(**BUILTIN_DATASETS[name].reader_params)
            except KeyError:
                raise ValueError('unknown reader ' + name +
                                 '. Accepted values are ' +
                                 ', '.join(BUILTIN_DATASETS.keys()) + '.')
        else:
            self.sep = sep
            self.skip_lines = skip_lines
            self.rating_scale = rating_scale

            lower_bound, higher_bound = rating_scale

            splitted_format = line_format.split()

            entities = ['user', 'item', 'rating']
            if 'timestamp' in splitted_format:
                self.with_timestamp = True
                entities.append('timestamp')
            else:
                self.with_timestamp = False

            # check that all fields are correct
            if any(field not in entities for field in splitted_format):
                raise ValueError('line_format parameter is incorrect.')

            self.indexes = [splitted_format.index(entity) for entity in
                            entities]

    def parse_line(self, line):
        """Parse a line.

        Ratings are translated so that they are all strictly positive.

        Args:
            line(str): The line to parse

        Returns:
            tuple: User id, item id, rating and timestamp. The timestamp is set
            to ``None`` if it does no exist.
            """

        line = line.split(self.sep)
        try:
            if self.with_timestamp:
                uid, iid, r, timestamp = (line[i].strip()
                                          for i in self.indexes)
            else:
                uid, iid, r = (line[i].strip()
                               for i in self.indexes)
                timestamp = None

        except IndexError:
            raise ValueError('Impossible to parse line. Check the line_format'
                             ' and sep parameters.')

        return uid, iid, float(r), timestamp
