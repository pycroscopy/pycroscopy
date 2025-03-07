from codecs import open
import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
readme_file = os.path.join(here, 'README.md')

if os.path.exists(readme_file):
    with open(readme_file, encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = ""

with open(os.path.join(here, 'pycroscopy/__version__.py')) as f:
    __version__ = f.read().split("'")[1]

# TODO: Move requirements to requirements.txt
requirements = ['numpy>=1.13.0',
                'scipy>=0.17,<1.10.2',
                'scikit-image>=0.12.3',
                'scikit-learn>=0.17.1',
                'matplotlib>=2.0.0',
                'torch>=1.0.0',
                'tensorly>=0.6.0',
                'tqdm',
                'ipywidgets>=5.2.2',
                'ipython',
                'simpleitk',
                'sidpy>=0.12.1',
                'pysptools',
                'cvxopt>=1.2.7',
                'SciFiReaders>=0.10.0',
                'pywget',
                'pyTEMlib>=0.2023.8.0',
                'ipympl'
                ]

setup(
    name='pycroscopy',
    version=__version__,
    description='Python library for scientific analysis of microscopy data',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
    test_suite='pytest',
    extras_require={},
    include_package_data=True,
)

