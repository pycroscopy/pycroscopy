from codecs import open
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst')) as f:
    long_description = f.read()

with open(os.path.join(here, 'pycroscopy/__version__.py')) as f:
    __version__ = f.read().split("'")[1]

# TODO: Move requirements to requirements.txt
requirements = ['numpy>=1.13.0',
                'scipy>=0.17.1',
                'scikit-image>=0.12.3',
                'scikit-learn>=0.17.1',
                'matplotlib>=2.0.0',
                'torch>=1.0.0',
                'tensorly>=0.6.0',
                'tqdm',
                'ipywidgets>=5.2.2',
                'ipython',
                'simpleitk',
                'sidpy>=0.0.6',
                'pysptools',
                'cvxopt>=1.2.7'
                ]

setup(
    name='pycroscopy',
    version=__version__,
    description='Python library for scientific analysis of microscopy data',
    long_description=long_description,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Information Analysis'],
    keywords=['EELS', 'STEM', 'TEM', 'XRD', 'AFM', 'SPM', 'STS', 'band excitation', 'BE', 'BEPS', 'Raman', 'NanoIR',
              'electron microscopy', ' scanning probe', ' x-rays', 
              'atomic force microscopy', 'SIMS', 'energy', 'spectroscopy', 'imaging', 'microscopy', 'spectra'
              'characterization', 'spectrogram', 'hyperspectral', 'multidimensional', 'data format', 'universal',
              'clustering', 'decomposition', 'curve fitting', 'data analysis', 'PCA', ' SVD', ' NMF', ' DBSCAN', ' kMeans',
              'machine learning', 'bayesian inference', 'fft filtering', 'signal processing', 'image cleaning',
              'denoising', 'model', 'msa', 'quantification'],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    url='https://pycroscopy.github.io/pycroscopy/about.html',
    license='MIT',
    author='Pycroscopy contributors',
    author_email='pycroscopy@gmail.com',
    install_requires=requirements,
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    platforms=['Linux', 'Mac OSX', 'Windows 10/8.1/8/7'],
    # package_data={'sample':['dataset_1.dat']}
    test_suite='pytest',
    extras_require={},
    # dependency='',
    # dependency_links=[''],
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
