from setuptools import setup, find_packages, Extension
from codecs import open
from os import path

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

__version__ = '1.0.1'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

cmdclass = {}

ext = '.pyx' if USE_CYTHON else '.c'

extensions = [Extension('surprise.similarities',
                       ['surprise/similarities' + ext],
                        include_dirs=[np.get_include()]),
              Extension('surprise.prediction_algorithms.matrix_factorization',
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
    ext_modules = cythonize(extensions)
    cmdclass.update({'build_ext': build_ext})
else:
	ext_modules = extensions

setup(
    name='scikit-surprise',
    author='Nicolas Hug',
    author_email='contact@nicolas-hug.com',

    description=('An easy-to-use library for recommender systems.'),
    long_description=long_description,

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
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 2.7',
    ],
    keywords='recommender recommendation system',

    packages=find_packages(exclude=['tests*']),
    include_package_data=True,
    ext_modules = ext_modules,
    cmdclass=cmdclass,
    install_requires=install_requires,
    dependency_links=dependency_links,

    entry_points={'console_scripts':
                 ['surprise = surprise.__main__:main']},
)
