from codecs import open
import os

on_rtd = os.environ.get('READTHEDOCS') == 'True'

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst')) as f:
    long_description = f.read()

if on_rtd:
    requirements = ['psutil', 'xlrd>=1.0.0']
else:
    requirements = ['numpy_groupies>=0.9.6', 'pyqtgraph>=0.10',
                    'h5py>=2.6.0', 'igor', 'matplotlib>=2.0.0',
                    'scikit-learn>=0.17.1', 'xlrd>=1.0.0','joblib>=0.11',
                    'psutil', 'scikit-image>=0.12.3', 'scipy>=0.17.1',
                    'numpy>=1.11.0', 'ipywidgets>=5.2.2', 'ipython>=5.1.0']

setup(
    name='pycroscopy',
    version='0.59.0',
    description='A suite of Python libraries for high performance scientific computing of microscopy data.',
    long_description=long_description,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        ],
    keywords='scientific microscopy data analysis',
    packages=find_packages(exclude='tests'),
    url='http://github.com/pycroscopy/pycroscopy',
    license='MIT',
    author='S. Somnath, C. R. Smith, N. Laanait',
    author_email='pycroscopy@gmail.com',

    # I don't remember how to do this correctly!!!. NL
    install_requires=requirements,
    # package_data={'sample':['dataset_1.dat']}
    test_suite='nose.collector',
    tests_require='Nose',
    dependency='',
    dependency_links=[''],
    include_package_data=True,

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    # package_data={
    #     'sample': ['package_data.dat'],
    # },

    # Although 'package_data' is the preferred approach, in some case you may
    # need to place data files outside of your packages. See:
    # http://docs.python.org/3.4/distutils/setupscript.html#installing-additional-files # noqa
    # In this case, 'data_file' will be installed into '<sys.prefix>/my_data'
    # data_files=[('my_data', ['data/data_file'])],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    # entry_points={
    #     'console_scripts': [
    #         'sample=sample:main',
    #     ],
    # },
)
