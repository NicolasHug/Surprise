"""
Module for testing the Reader class.
"""

from pyrec import Reader

def test_parse_line():
    """Test the parse_line method"""

    # Basic line parsing
    line_format = 'user item rating timestamp'
    sep = ','
    reader = Reader(line_format=line_format, sep=sep)

    line = 'me,best_movie_ever, 5 ,25111990'
    uid, iid, rating, timestamp = reader.parse_line(line)

    assert uid == 'me'
    assert iid == 'best_movie_ever'
    assert rating == 5
    assert timestamp == '25111990'

    # Change order of fields (and sep)
    line_format = 'timestamp rating item user'
    sep = ' '
    reader = Reader(line_format=line_format, sep=sep)

    line = '25111990 5 best_movie_ever me'
    uid, iid, rating, timestamp = reader.parse_line(line)

    assert uid == 'me'
    assert iid == 'best_movie_ever'
    assert rating == 5
    assert timestamp == '25111990'

    # Without timestamp (changed sep as well)
    line_format = 'rating item user'
    sep = '-'
    reader = Reader(line_format=line_format, sep=sep)

    line = '5 - best_movie_ever - me'
    uid, iid, rating, _ = reader.parse_line(line)

    assert uid == 'me'
    assert iid == 'best_movie_ever'
    assert rating == 5
