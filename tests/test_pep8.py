"""
Module for testing if the code is PEP8 compliant.
"""

from logging import getLogger

from flake8.api import legacy as flake8


getLogger('flake8').propagate = False  # disable annoying flake8 DEBUG messages


def test_regular_files():

    style_guide = flake8.get_style_guide(
        filename=['*.py'],
        exclude=['doc', '.eggs', '*.egg', 'build', 'benchmark.py',
                 'run_all_examples.py'],
        select=['E', 'W', 'F'],
        max_line_length=88
    )

    report = style_guide.check_files()

    assert report.get_statistics('E') == []
    assert report.get_statistics('W') == []
    assert report.get_statistics('F') == []


def test_cython_files():

    style_guide = flake8.get_style_guide(
        filename=['*.pyx', '*.px'],
        exclude=['doc', '.eggs', '*.egg', 'build', 'setup.py'],
        select=['E', 'W', 'F'],
        ignore=['E225'],
        max_line_length=88
    )

    report = style_guide.check_files()

    assert report.get_statistics('E') == []
    assert report.get_statistics('W') == []
    assert report.get_statistics('F') == []
