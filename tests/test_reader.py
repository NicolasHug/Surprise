"""
Module for testing the Reader class.
"""


import pytest

from surprise import Reader


def test_params():
    """Test Reader parameters"""

    with pytest.raises(ValueError):
        Reader(name="wrong_name")

    with pytest.raises(ValueError):
        Reader(line_format="users item rating")

    with pytest.raises(ValueError):
        Reader(line_format="user itemm rating")

    with pytest.raises(ValueError):
        Reader(line_format="item user rrating")

    with pytest.raises(ValueError):
        Reader(line_format="item BLABLA user rating")


def test_parse_line():
    """Test the parse_line method"""

    # Basic line parsing
    line_format = "user item rating timestamp"
    sep = ","
    reader = Reader(line_format=line_format, sep=sep)

    line = "me,best_movie_ever, 5 ,25111990"
    uid, iid, rating, timestamp = reader.parse_line(line)

    assert uid == "me"
    assert iid == "best_movie_ever"
    assert rating == 5
    assert timestamp == "25111990"

    # Change order of fields (and sep)
    line_format = "timestamp rating item user"
    sep = " "
    reader = Reader(line_format=line_format, sep=sep)

    line = "25111990 5 best_movie_ever me"
    uid, iid, rating, timestamp = reader.parse_line(line)

    assert uid == "me"
    assert iid == "best_movie_ever"
    assert rating == 5
    assert timestamp == "25111990"

    # Without timestamp (changed sep as well)
    line_format = "rating item user"
    sep = "-"
    reader = Reader(line_format=line_format, sep=sep)

    line = "5 - best_movie_ever - me"
    uid, iid, rating, _ = reader.parse_line(line)

    assert uid == "me"
    assert iid == "best_movie_ever"
    assert rating == 5

    # Wrong sep
    line_format = "rating item user"
    sep = ";"
    reader = Reader(line_format=line_format, sep=sep)

    line = "5 - best_movie_ever - me"
    with pytest.raises(ValueError):
        uid, iid, rating, _ = reader.parse_line(line)

    # Wrong number of fields
    line = "5 - best_movie_ever"
    with pytest.raises(ValueError):
        uid, iid, rating, _ = reader.parse_line(line)
