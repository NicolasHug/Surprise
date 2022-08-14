#!/bin/sh

set -ex

black --version  # 22.6.0 (on Python 3.9)
usort --version  # 1.0.4
flake8 --version  # 5.0.4

flake8 --max-line-length 88 --ignore E203,E402,W503,W504,F821,E501 surprise
flake8 --max-line-length 88 tests
flake8 --max-line-length 88 examples

usort format surprise
usort format tests
usort format examples

black surprise
black tests
black examples
