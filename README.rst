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
   1. Install Anaconda 2.5.0 Python 2.7 64-bit:
      
      a. Mac users: <https://repo.continuum.io/archive/Anaconda2-2.5.0-MacOSX-x86_64.pkg>
      
      b. Windows users: <https://repo.continuum.io/archive/Anaconda2-2.5.0-Windows-x86_64.exe>

      c. Linux users: <https://repo.continuum.io/archive/Anaconda2-2.5.0-Linux-x86_64.sh>
      
   2. Install a C compiler - Windows users ONLY:
      
      If you are running on a Windows machine, you will need to load a C compiler, which is required to build certain packages in python including one for parallel processing. Mac Users can skip this step as OS X natively comes with a C compiler.
      
      a. Install Microsoft Visual C++ 2008 SP1 Redistributable Package (x64) from <http://www.microsoft.com/en-us/download/confirmation.aspx?id=2092>
      
      b. Install Microsoft Visual C++ for Python from <http://aka.ms/vcpython27>
      
   3. Install multiprocess for parallel computation tasks
   
      a. Max or Linux users: 
      
         Open a terminal and type: 
            
            pip install multiprocess
         
      b. Windows users:
      
         * Download multiprocess from <https://pypi.python.org/pypi/multiprocess#downloads> and extract to a folder
         
         * Open the Command Prompt by going to Start >> Run >> type:
            
            cmd.exe
         
         * Navigate to where the folder was extracted
         
         * Type:
         
               python setup.py build
               
               python setup.py install
               
   4. Install pycroscopy:
   
      open a terminal (mac / linux) or command prompt (windows) and type:
      
         pip install pycroscopy
         
   5. Enjoy pycroscopy!
               
         
