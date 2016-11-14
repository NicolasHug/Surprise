from setuptools import setup, find_packages, Extension
from codecs import open
from os import path

__version__ = '0.0.4'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]


### Cython stuff

from Cython.Build import cythonize
import numpy as np

extensions = [Extension('recsys.similarities',
                        ['recsys/similarities.pyx'],
                        include_dirs=[np.get_include()]),
              Extension('recsys.prediction_algorithms.matrix_factorization',
                        ['recsys/prediction_algorithms/matrix_factorization.pyx'],
                        include_dirs=[np.get_include()]),
             ]
ext_modules = cythonize(extensions)

setup(
    name='recsys',
    version=__version__,
    description=('A recommender system package aimed towards researchers ' +
                 'and students.'),
    long_description=long_description,
    url='https://github.com/Niourf/recsys',
    download_url='https://github.com/Niourf/recsys/tarball/' + __version__,
    license='GPLv3+',
    classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Developers',
      'Intended Audience :: Education',
      'Intended Audience :: Science/Research',
      'Topic :: Scientific/Engineering',
      'License :: OSI Approved',
      'Programming Language :: Python :: 3',
    ],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,

    ext_modules = ext_modules,

    author='Nicolas Hug',
    install_requires=install_requires,
    dependency_links=dependency_links,
    author_email='nh.nicolas.hug@gmail.com'
)
