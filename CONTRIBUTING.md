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


All the tools needed for the development of Surprise (sphinx, flake8,
etc...) can be installed by running

    pip install -r requirements_dev.txt

Then, you can install your local copy of the repo by running

    pip install -e .

Running tests
-------------

We use [pytest](http://doc.pytest.org/en/latest/) so simply running

    pytest

in the root directory should do the job.

Check coding style
------------------

You can check that your code is PEP8 compliant by running

    pytest tests/test_pep8.py

Building the docs locally
-------------------------

The docs can be compiled with

    cd doc
    make html

You can check the results in `doc/build/html`. Please make sure that the docs
compile without errors. Run `make clean` from time to time in order to avoid
hidden warnings. You can check spelling mistakes by running

    make spelling

Legit words that are not recognized can be added in the
`source/spelling_wordlist.txt` file.
