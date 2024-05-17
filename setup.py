from setuptools import Extension, setup

import numpy as np
from Cython.Build import cythonize

"""
Release instruction:

Update changelog and contributors list. If you ever change the dependencies in
`pyproject.toml`, also update the hardcoded numpy version under [build-system] -> requires

Basic local checks:
- tests run correctly
- doc compiles without warning (make clean first).

Check that the latest RTD build was OK: https://readthedocs.org/projects/surprise/builds/

Change __version__ in setup.py to new version name. Also update the hardcoded
version in build_sdist.yml, otherwise the GA jobs will fail.

The sdist is built on 3.8 by GA:
- check the sdist building process. It should compile pyx files and the C files
  should be included in the archive
- check the install jobs. Look for compilation warnings. Make sure Cython isn't
  needed and only C files are compiled.
- check test jobs for warnings etc.

It's best to just get the sdist artifact from the job instead of re-building it
locally. Get the "false" sdist: false == with `numpy>=` constraint, not with
`oldest-supported-numpy`. We don't want `oldest-supported-numpy` as the uploaded
sdist because it's more restrictive.

Then upload to test pypi:
    twine upload blabla.tar.gz -r testpypi

Check that install works on testpypi, then upload to pypi and check again.
to install from testpypi:
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple scikit-surprise  # noqa
Doesn't hurt to check that the tests pass after installing from testpypi.

If not already done, sync gh-pages with the master's README

push new release tag on github (commit last changes first if needed):
    git tag vX.Y.Z
    git push --tags

Check that RTD has updated 'stable' to the new release (may take a while).

In the mean time, upload to conda:
    - Compute SHA256 hash of the new .tar.gz archive (or check it up on PyPI)
    - update recipe/meta.yaml on feedstock fork consequently (only version and
      sha should be changed.  Maybe add some import tests).
    - Push changes, Then open pull request on conda-forge feedstock and merge it
      when all checks are OK. Access the conda-forge feedstock it by the link on
      GitHub 'forked from blah blah'.
    - Check on https://anaconda.org/conda-forge/scikit-surprise that new
      version is available for all platforms.

Then, maybe, celebrate.
"""

# This prevents Cython from using deprecated numpy C APIs
define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

# We're using numpy C APIs in our Cython code so Cython will generate C code
# that requires the numpy headers. We need to tell the compiler where to find
# those headers.
# If you remove this and compilation still works, don't get fooled: it's
# probably only because the numpy headers are available in the default locations
# like /usr/include/numpy/ so they get found. But that wouldn't necessarily be
# the case on other users' machines building the sdist.
include_dirs = [np.get_include()]

extensions = [
    Extension(
        # name is where the .so will be placed, i.e. where the module will be
        # importable from.
        name="surprise.similarities",
        sources=["surprise/similarities.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
    ),
    Extension(
        name="surprise.prediction_algorithms.matrix_factorization",
        sources=["surprise/prediction_algorithms/matrix_factorization.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
    ),
    Extension(
        name="surprise.prediction_algorithms.optimize_baselines",
        sources=["surprise/prediction_algorithms/optimize_baselines.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
    ),
    Extension(
        name="surprise.prediction_algorithms.slope_one",
        sources=["surprise/prediction_algorithms/slope_one.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
    ),
    Extension(
        name="surprise.prediction_algorithms.co_clustering",
        sources=["surprise/prediction_algorithms/co_clustering.pyx"],
        include_dirs=include_dirs,
        define_macros=define_macros,
    ),
]

extensions = cythonize(
    extensions,
    compiler_directives={
        "language_level": 3,
        "boundscheck": False,
        "wraparound": False,
        "initializedcheck": False,
        "nonecheck": False,
    },
)

setup(ext_modules=extensions)
