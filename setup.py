from setuptools import setup, find_packages, Extension
from codecs import open
from os import path

"""
Release instruction:

Upate changelog and contributors list.

Check that tests run correctly for 36 and 27 and doc compiles without warning
(make clean first).

change __version__ in setup.py to new version name.

First upload to test pypi:
    mktmpenv (Python version should not matter)
    pip install numpy cython twine
    python setup.py sdist
    twine upload dist/blabla.tar.gz -r testpypi

Check that install works on testpypi, then upload to pypi and check again.
to install from testpypi:
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple scikit-surprise  # noqa
Doesn't hurt to check that the tests pass after installing from testpypi.

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

from setuptools import dist  # Install numpy right now
dist.Distribution().fetch_build_eggs(['numpy>=1.11.2'])

try:
    import numpy as np
except ImportError:
    exit('Please install numpy>=1.11.2 first.')

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

__version__ = '1.1.1'

here = path.abspath(path.dirname(__file__))

# Get the long description from README.md
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '')
                    for x in all_reqs if x.startswith('git+')]

cmdclass = {}

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [
    Extension(
        'surprise.similarities',
        ['surprise/similarities' + ext],
        include_dirs=[np.get_include()]
    ),
    Extension(
        'surprise.prediction_algorithms.matrix_factorization',
        ['surprise/prediction_algorithms/matrix_factorization' + ext],
        include_dirs=[np.get_include()]),
    Extension('surprise.prediction_algorithms.optimize_baselines',
              ['surprise/prediction_algorithms/optimize_baselines' + ext],
              include_dirs=[np.get_include()]),
    Extension('surprise.prediction_algorithms.slope_one',
              ['surprise/prediction_algorithms/slope_one' + ext],
              include_dirs=[np.get_include()]),
    Extension('surprise.prediction_algorithms.co_clustering',
              ['surprise/prediction_algorithms/co_clustering' + ext],
              include_dirs=[np.get_include()]),
]

if USE_CYTHON:
    ext_modules = cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
        },
    )
    cmdclass.update({'build_ext': build_ext})
else:
    ext_modules = extensions

setup(
    name='scikit-surprise',
    author='Nicolas Hug',
    author_email='contact@nicolas-hug.com',

    description=('An easy-to-use library for recommender systems.'),
    long_description=long_description,
    long_description_content_type='text/markdown',

    version=__version__,
    url='http://surpriselib.com',

    license='GPLv3+',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='recommender recommendation system',

    packages=find_packages(exclude=['tests*']),
    python_requires=">=3.7",
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
    dependency_links=dependency_links,

    entry_points={'console_scripts':
                  ['surprise = surprise.__main__:main']},
)
