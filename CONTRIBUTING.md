Contributing to Surprise
========================

Pull requests are always welcome! Before submitting a new pull request, please
make sure that:

* Your code is [clean](https://www.youtube.com/watch?v=wf-BqAjZb8M),
  [pythonic](https://www.youtube.com/watch?v=OSGv2VnC0go), well commented and
  if it's a new feature/algorithm, that it's also well documented.
* Your code passes the tests (we use
  [pytest](http://doc.pytest.org/en/latest/)), plus the ones that you wrote ;)
* Your code is [PEP 8](https://www.python.org/dev/peps/pep-0008/) compliant.
  The bare minimum is that
  [flake8](http://flake8.pycqa.org/en/latest/index.html) does not report any
  warning (see below).


All the tools needed for the development of surprise (sphinx, flake8,
etc...) can be installed by running

    pip install -r requirements_dev.txt

Running tests
-------------

We use [pytest](http://doc.pytest.org/en/latest/) so simply running

    pytest

in the root directory should do the job.

Check coding style with flake8
------------------------------

Please make sure that your code is PEP8 compliant by running

    flake8

If you wrote any Cython code, also use

    flake8 --config .flake8.cython

Building the docs locally
-------------------------

The docs can be compiled with

    cd doc
    make html

You can check the results in `doc/build/html`. Please make sure that the docs
compile without errors. Run `make clean` from time to time in order to avoid
hidden warnings.
