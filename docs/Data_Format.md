
# Developing Scientific Workflows in Pycroscopy - Part 0: Data Format

#### Suhas Somnath
8/8/2017

This set of notebooks will serve as examples for developing and end-to-end workflows for and using pycroscopy. 

This preliminary document goes over the pycroscopy data format

## Why should you care?

The quest for understanding more about samples has necessitated the development of a multitude of microscopes, each capable of numerous measurement modalities. 

Typically, each commercial microscope generates data files formatted in proprietary data formats by the instrument manufacturer. The proprietary natures of these data formats impede scientific progress in the following ways:
1. By making it challenging for researchers to extract data from these files 
2. Impeding the correlation of data acquired from different instruments.
3. Inability to store results back into the same file
4. Accomodating files from few kilobytes to several gigabytes of data
5. Requiring different versions of analysis routines for each format

Future concerns:
1. Several fields are moving towards the open science paradigm which will require journals and researchers to support journal papers with data and analysis software 
2. US Federal agencies that support scientific research require curation of datasets in a clear and organized manner

To solve these and many more problems, we have developed an __instrument agnostic data format__ that can be used to represent data from any instrument, size, dimensionality, or complexity.

## Pycroscopy data format

Regardless of origin, modality or complexity, imaging data have one thing in common:

__The same measurement is performed at multiple spatial locations__

The data format in pycroscopy is based on this one simple ground truth. The data always has some spatial dimensions (X, Y, Z) and some spectroscopic dimensions (time, frequency, intensity, wavelength, temperature, cycle, voltage, etc.). In pycroscopy, the spatial dimensions are collapsed onto a single dimension and the spectroscopic dimensions are flattened to the other dimensions. Thus, all data are stored as two dimensional grids. Here are some examples of how some familar data can be represented using this paradigm:
* __Grayscale photographs__: A single value (intensity) in is recorded at each pixel in a two dimensional grid. Thus, there are are two spatial dimensions - X, Y and one spectroscopic dimension - "Intensity". The data can be represented as a N x 1 matrix where N is the product of the number of rows and columns of pixels. The second axis has size of 1 since we only record one value (intensity) at each location. __The positions will be arranged as row0-col0, row0-col1.... row0-colN, row1-col0....__ Color images or photographs will be discussed below due to some very important subtleties about the measurement.
* A __single Raman spectra__: In this case, the measurement is recorded at a single location. At this position, data is recorded as a function of a single (spectroscopic) variable such as wavelength. Thus this data is represented as a 1 x P matrix, where P is the number of points in the spectra
* __Scanning Tunelling Spectroscopy or IV spectroscopy__: The current (A 1D array of size P) is recorded as a function of voltage at each position in a two dimensional grid of points (two spatial dimensions). Thus the data would be represente as a N x P matrix, where N is the product of the number of rows and columns in the grid and P is the number of spectroscopic points recorded. 
    * If the same voltage sweep were performed twice at each location, the data would be represented as N x 2 P. The data is still saved as a long (2*P) 1D array at each location. The number of spectroscopic dimensions would change from just ['Voltage'] to ['Voltage', 'Cycle'] where the second spectroscopic dimension would account for repetitions of this bias sweep.
        * __The spectroscopic data would be stored as it would be recorded as volt_0-cycle_0, volt_1-cycle_0..... volt_P-1-cycle_0, volt_0-cycle_1.....volt_P-1-cycle-1. Just like the positions__
    * Now, if the bias was swept thrice from -1 to +1V and then thrice again from -2 to 2V, the data bacomes N x 2 * 3 P. The data now has two position dimensions (X, Y) and three spectrosocpic dimensions ['Voltage', 'Cycle', 'Step']. The data is still saved as a (P * 2 * 3) 1D array at each location. 
    
### Compound Datasets:
pycroscopy actually uses compound datasets a lot more frequently than one would think. The need and utility of compound datasets are best described with examples: 
* __Color images__: Each position in these datasets contain three (red, blue, green) or four (cyan, black, magenta, yellow) values. One would naturally be tempted to simply treat these datasets as N x 3 datasets and it certainly is not wrong to represent data this way. However, storing the data in this format would mean that the red intensity was collected first, followed by the green, and finally by the blue. In other words, a notion of chronology is attached to both the position and spectroscopic axis if one strictly follows the pycroscopy defenition. While the intensities for each color may be acquired sequentially in detectors, we will assume that they are acquired simultaneously for this argument. In these cases, we store data using __compound datasets__ that allow the storage of multiple pieces of data within the same cell. While this may seem confusing or implausible, remember that computers store complex numbers in the same way. The complex numbers have a 'real' and a 'imaginary' component just like color images have 'red', 'blue', and 'green' components that describe a single pixel. Therefore, color images in pycroscopy would be represented by a N x 1 matrix with compound values instead of a N x 3 matrix with real or integer values. One would refer to the red component at a particular position as `dataset[position_index, spectroscopic_index]['red']`.
* __Functional fits__: Let's take the example of a N x P dataset whose spectra at each location are fitted to a complicated equation. Now the P points in the spectra will be represented by S  coefficients that don't necessarily follow any order. Consequently, the result of the functional fit should actually be a N x 1 dataset where each element is a compound value made up of the S coefficients. Note that while some form of sequence can be forced onto the coefficients if the spectra were fit to polynomial equations, the drawbacks outweight the benefits:
    * Storing data in compund datasets circumvents (slicing) problems associated with getting a specific / the kth coeffient if the data were stored in a real-valued matrix instead. 
    * Visualization also becomes a lot simpler since compound datasets cannot be plotted without specifying the component / coefficient of interest. This avoids plots with alternating coefficients that are several orders of magnitude larger / smaller than each other.   
     
### Making sense of such flattened datasets:
Each main dataset is always accompanied by four ancillary datasets: 
* the position value and index of each spatial location (row)
* the spectroscopic value and index of any column in the dataset
In addition to serving as a legend or the key, these ancillary datasets are necessary for explaining:
* the original dimensionality of the dataset
* how to reshape the data back to its N dimensional form

From the __IV Spectorscopy__ example with [X, Y] x [Voltage, Cycle, Step]:
* The position datasets would be of shape N x 2 - N total position, two spatial dimensions. 
    * The position indices datasets may start like: 
    
| 0 | 0 |
| 0 | 1 |
| a | t |


        * 0, 0
        * 0, 1
        * ....
        * 0, N/2
        * 1, 0 ....
        would be structured exactly

#### Channels
The pycroscopy data format also allows multiple channels of information to be recorded as separate datasets in the same file. For example, one channel could be a spectra (1D array) collected at each location on a 2D grid while another could be the temperature (single value) recorded by another sensor at the same spatial positions. In many cases, multiple channels of information, such as strain, current, etc., are acquired for each position in the dataset. In these cases, all the ancillary datasets are repeated / referenced. They can have different spectroscopic / positions. 

My hope is that this notebook will serve as a comprehensive example for:
  
1. __Data Access__
    1. Loading, reading, writing, and manipulating HDF5 / H5 files.
            
2. __Visualization__
    1. Visualizing results of analyses and processing using pycroscopy functions
    2. Developing simple interactive visualizers

Among the numerous benefits of __HDF5__ files are that these files:
* are readily compatible with high-performance computing facilities
* scale very efficiently from few kilobytes to several terabytes
* can be read and modified using any language including Python, Matlab, C/C++, Java, Fortran, Igor Pro, etc.

# NEED TO SAY THAT ALL THESE ANCILLARY MATRICES NEED TO BE 2 DIMENSIONAL   

# HAVE NOT TALKED ABOUT DIMENSIONALITY OF THESE DATASETS AND WHAT THAT MEANS

# HAVE NOT SPOKEN ABOUT REGION REFERENCES

# HAVE NOT SPOKEN ABOUT MANDATORY ATTRIBUTES

# DATA GROUP NOMENCLATURE AND ATTRIBUTES STANDARDS

# ABILITY TO PERFORM THE SAME OPERATION MULTIPLE TIMES

* pycroscopy uses the __heirarchical data format (HDF5)__ files to store data
* These HDF5 or H5 files contain datasets and datagroups
* pycroscopy data files have two kinds of datasets:
    * __`main`__ datasets: These must be of the form: `[instance, features]`. 
        * All imaging or measurement data satisfy this category, where positions form the instances and the spectral points form the features. Thus, even standard 2D images or a single spectra also satisfy this condition.
        * A collection of `k` chosen spectra would still satisfy this condition. Some examples include:
            * the cluster centers obtained from a clustering algorithm like `k-Means clustering`.
            * The abundance maps obtained from decomposition algorithms like `Singular Value Decomposition (SVD)` or `Non-negetive matrix factorization (NMF)`
    * __`ancillary`__ datasets: All other datasets fall into this category. These include the frequency vector or bias vector as a function of which the main dataset was collected.
* pycroscopy stores all data in two dimensional matrices with all position dimensions collapsed to the first dimension and all other (spectroscopic) dimensions collapsed to the second dimension. 
* All these __`main`__ datasets are always accompanied by four ancillary datasets:
    * Position Indices
    * Position Values
    * Spectroscopic Indices
    * Spectroscopic Values
      
* All __`main`__ datasets always have two attributes that describe the measurement itself:
    * `quantity`: The physical quantity contained in each cell of the dataset - such as voltage, current, force etc.
    * `units`: The units for the physical quantity such as `V` for volts, `nA` for nano amperes, `pN` for pico newtons etc.
* All __`main`__ datasets additionally have 4 attributes that provide the references or links to the 4 aforementions ancillary datasets
    * Storing just the references allows us to re-use the same position / spectroscopic datasets without having to remake them
    
This bookkeeping is necesary for helping the code to understand the dimensionality and structure of the data. While these rules may seem tedious, there are several functions and a few classes that make these tasks much easier


```python

```
