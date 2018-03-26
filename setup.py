from codecs import open
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst')) as f:
    long_description = f.read()

requirements = ['numpy_groupies>=0.9.6',
                'pyqtgraph>=0.10',
                'h5py>=2.6.0',
                'igor',
                'matplotlib>=2.0.0',
                'scikit-learn>=0.17.1',
                'xlrd>=1.0.0',
                'joblib>=0.11.0',
                'psutil',
                'scikit-image>=0.12.3',
                'scipy>=0.17.1',
                'numpy>=1.11.0',
                'ipywidgets>=5.2.2',
                'ipython>=5.1.0']

setup(
    name='pycroscopy',
    version='0.59.dev6',
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
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Information Analysis'],
    keywords=['EELS', 'STEM', 'TEM', 'XRD', 'AFM', 'SPM', 'STS', 'band excitation', 'BE', 'BEPS', 'Raman', 'NanoIR',
              'ptychography', 'g-mode', 'general mode', 'electron microscopy', ' scanning probe', ' x-rays', 'probe',
              'atomic force microscopy', 'SIMS', 'energy', 'spectroscopy', 'imaging', 'microscopy', 'spectra'
              'characterization', 'spectrogram', 'hyperspectral', 'multidimensional', 'data format', 'universal',
              'clustering', 'decomposition', 'curve fitting', 'data analysis PCA', ' SVD', ' NMF', ' DBSCAN', ' kMeans',
              'machine learning', 'bayesian inference', 'fft filtering', 'signal processing', 'image cleaning',
              'denoising', 'model', 'msa', 'quantification',
              'png', 'tiff', 'hdf5', 'igor', 'ibw', 'dm3', 'oneview', 'KPFM', 'FORC', 'ndata',
              'Asylum', 'MFP3D', 'Cypher', 'Omicron', 'Nion', 'Nanonis', 'FEI'],
    packages=find_packages(exclude='tests'),
    url='https://pycroscopy.github.io/pycroscopy/about.html',
    license='MIT',
    author='S. Somnath, C. R. Smith, N. Laanait',
    author_email='pycroscopy@gmail.com',
    install_requires=requirements,
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'Nose'],
    platforms=['Linux', 'Mac OSX', 'Windows 10/8.1/8/7'],
    # package_data={'sample':['dataset_1.dat']}
    test_suite='nose.collector',
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
