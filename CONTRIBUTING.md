Contributing to Surprise
========================

Pull requests are always welcome! Before submitting a new pull request, please
make sure that:

* your code is clean, well commented and if it's a new feature/algorithm, that
  it's also well documented.
* your code passes the tests (use [pytest](http://doc.pytest.org/en/latest/)),
  plus the ones that you wrote ;)
* your code is [PEP 8](https://www.python.org/dev/peps/pep-0008/) compliant.
  You can use [flake8](http://flake8.pycqa.org/en/latest/index.html) for
  checking.

Running tests
-------------

Simply running

    pytest

in the root directory should  do the job

Building the docs locally
-------------------------

To check your changes to the documentation, you can build the docs locally. If you haven't already, install all the packages for development with

`pip install -r requirements_dev.txt`

this will install [sphinx](http://www.sphinx-doc.org/en/1.5.1/), the RTD theme and some sphinx extensions needed in the doc building process.The you should be able to build the docs locally by running

    cd doc
    make html

You can check the results in `doc/build/html`.
