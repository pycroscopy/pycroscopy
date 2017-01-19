pycroscopy
==========

See <https://pycroscopy.github.io/pycroscopy/> for more info.

0. Description
--------------
A python package for image processing and scientific analysis of imaging modalities such as multi-frequency scanning probe microscopy,
scanning tunneling spectroscopy, x-ray diffraction microscopy, and transmission electron microscopy.
Classes implemented here are ported to a high performance computing platform at Oak Ridge National Laboratory (ORNL).

1. Package Structure
--------------------
The package structure is simple, with 4 main modules:
   1. `io`: Input/Output from custom & proprietary microscope formats to HDF5.
   2. `processing`: Multivariate Statistics, Machine Learning, and Filtering.
   3. `analysis`: Model-dependent analysis of image information.
   4. `viz`: Visualization and interactive slicing of high-dimensional data by lightweight Qt viewers.

Once a user converts their microscope's data format into an HDF5 format, by simply extending some of the classes in `io`, the user gains access to the rest of the utilities present in `pycroscopy.*`. 

2. Installation
---------------
Pycroscopy requires the installation of a development environment such as Spyder from Continuum or PyCharm. 

   1. Uninstall existing Python 2.7 distribution(s) if installed.  Restart computer afterwards.

   2. Install Anaconda 4.2.13 Python 2.7 64-bit:
      
      a. Mac users: <https://repo.continuum.io/archive/Anaconda2-4.2.0-MacOSX-x86_64.pkg>
      
      b. Windows users: <https://repo.continuum.io/archive/Anaconda2-4.2.0-Windows-x86_64.exe>

      c. Linux users: <https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh>
	  
   3. Install pycroscopy:
   
      Open a terminal (mac / linux) or command prompt (windows, if possible with administrator priveleges) and type:
      
         pip install pycroscopy
         
   4. Enjoy pycroscopy!
