"""
Created on 7/17/16 10:08 AM
@author: Numan Laanait
"""

from warnings import warn

import numpy as np
import psutil

from .guess_methods import GuessMethods
from ..io.hdf_utils import checkIfMain,getAuxData
from ..io.io_hdf5 import ioHDF5


class Model(object):
    """
    Encapsulates the typical routines performed during model-dependent analysis of data.
    This abstract class should be extended to cover different types of imaging modalities.

    """
    def __init__(self, h5_main, variables=['Frequency']):
        """
        For now, we assume that the guess dataset has not been generated for this dataset but we will relax this requirement
        after testing the basic components.

        Parameters:
        ----
        h5_main : h5py.Dataset instance
            The dataset over which the analysis will be performed. This dataset should be linked to the spectroscopic
            indices and values, and position indices and values datasets.
        variables : list(string), Default ['Frequency']
            Lists of attributes that h5_main should possess so that it may be analyzed by Model.
        Returns:
        -------
        None
        """
        # Checking if dataset is "Main"
        if self.__isLegal(h5_main, variables):
            self.h5_main = h5_main
            self.hdf = ioHDF5(self.h5_main.file)

        else:
            warn('Provided dataset is not a "Main" dataset with necessary ancillary datasets')
            return
        # Checking if parallel processing will be used
        try:
            import multiprocess
            self.__parallel = True
        except ImportError:
            warn("Multiprocess package (pip,github) is needed for parallel computation.\nSwitching to serial version.")
            self.__parallel = False

        # Determining the max size of the data that can be put into memory
        self.__setMemoryAndCPUs()


    def __setMemoryAndCPUs(self):
        """
        Checks hardware limitations such as memory, # cpus and sets the recommended datachunk sizes and the
        number of cores to be used by analysis methods.

        Returns
        -------
        None
        """

        if self.__parallel:
            self.__maxCpus = psutil.cpu_count() - 1
        else:
            self.__maxCpus = 1
        self.__maxMemoryMB = psutil.virtual_memory().available/1e6 # in MB

        self.__maxDataChunk = self.__maxMemoryMB/self.__maxCpus




    def __isLegal(self, h5_main, variables):
        """
        Checks whether or not the provided object can be analyzed by this Model class.
        Classes that extend this class will do additional checks to ensure that the supplied dataset is legal.

        Parameters:
        ----
        h5_main : h5py.Dataset instance
            The dataset over which the analysis will be performed. This dataset should be linked to the spectroscopic
            indices and values, and position indices and values datasets.

        variables : list(string)
            The dimensions needed to be present in the attributes of h5_main to analyze the data with Model.
        Returns:
        -------
        legal : Boolean
            Whether or not this dataset satisfies the necessary conditions for analysis
        """

        # Check if h5_main is a "Main" dataset
        cond_A = checkIfMain(h5_main)

        # Check if variables are in the attributes of spectroscopic indices
        h5_spec_vals = getAuxData(h5_main, auxDataName=['Spectroscopic_Values'])[0]
        # assert isinstance(h5_spec_vals, list)
        cond_B =  set(variables).issubset(set(h5_spec_vals.attrs.keys()))

        if cond_A and cond_B:
            legal = True
        else:
            legal = False

        return legal


    def __getDataChunk(self):
        """
        Returns a chunk of data for the guess or the fit

        Parameters:
        -----
        None

        Returns:
        --------
        dset : n dimensional array
            A portion of the main dataset
        """
        self.data = None

    def __getGuessChunk(self):
        """
        Returns a chunk of guess dataset corresponding to the main dataset

        Parameters:
        -----
        None

        Returns:
        --------
        dset : n dimensional array
            A portion of the guess dataset
        """
        self.guess = None

    def __setDataChunk(self, data_chunk, is_guess=False):
        """
        Writes the provided chunk of data into the guess or fit datasets. This method is responsible for any and all book-keeping

        Parameters
        ---------
        data_chunk : nd array
            n dimensional array of the same type as the guess / fit dataset
        is_guess : Boolean
            Flag that differentiates the guess from the fit
        """
        pass

    def __createGuessDatasets(self):
        """
        Model specific call that will write the h5 group, guess dataset, corresponding spectroscopic datasets and also
        link the guess dataset to the spectroscopic datasets. It is recommended that the ancillary datasets be populated
        within this function.

        The guess dataset will NOT be populated here but will be populated by the __setData function
        The fit dataset should NOT be populated here unless the user calls the optimize function.

        Parameters
        --------
        None

        Returns
        -------
        None
        """
        self.guess = None # replace with actual h5 dataset
        pass

    def computeGuess(self, strategy='Wavelet_Peaks', options={"peaks_widths": np.array([10,200])}, **kwargs):
        """

        Parameters
        ----------
        data
        strategy: string
            Default is 'Wavelet_Peaks'.
            Can be one of ['wavelet_peaks', 'relative_maximum', 'gaussian_processes']. For updated list, run GuessMethods.methods
        options: dict
            Default {"peaks_widths": np.array([10,200])}}.
            Dictionary of options passed to strategy. For more info see GuessMethods documentation.

        kwargs:
            processors: int
                number of processors to use. Default all processors on the system except for 1.

        Returns
        -------

        """

        self.__createGuessDatasets()

        processors = kwargs.get("processors", self.__maxCpus)
        gm = GuessMethods()
        if strategy["method"] in gm.methods:
            func = gm.__getattribute__(strategy["method"])(strategy["kwargs"])
            if self.__parallel:
                # start pool of workers
                print('launching %i kernels...'%(processors))
                pool = multiprocess.Pool(processors)
                # TODO: this is where we do the following
                """
                read first chunk
                while chunk is not empty:
                call the actual guess function on this data
                write the guess to the H5 dataset
                request for next chunk
                """
                tasks = [(vector) for vector in self.data]
                chunk = int(self.data.shape[0]/processors)
                jobs = pool.imap(func, tasks, chunksize = chunk)

                # get peaks from different processes
                results =[]
                print('Extracting Peaks...')
                try:
                    for j in jobs:
                        results.append(j)
                except ValueError:
                    print('Error: ValueError something about 2d- image. Probably one of the ORB input params are wrong')
                self.guesses = results
            else:
                results = np.array([ func(vec) for vec in vector])
        else:
            warn('Error: %s is not implemented in pycroscopy.analysis.GuessMethods to find guesses' %(strategy))




    def __createFitDataset(self):
        """
        Model specific call that will write the HDF5 fit dataset. pycroscopy requires that the h5 group, guess dataset,
        corresponding spectroscopic and position datasets be created and populated at this point.
        This function will create the HDF5 dataset for the fit and link it to same ancillary datasets as the guess.
        The fit dataset will NOT be populated here but will instead be populated using the __setData function

        Parameters
        --------
        None

        Returns
        -------
        None
        """
        self.fit = None  # replace with actual h5 dataset
        pass

    def computeFit(self):
        """
        Generates the fit for the given dataset and writes back to file

        Parameters
        ---------
        None

        Returns
        ----------
        None
        """
        if not self.guess is None:
            print("You need to guess before fitting")
            return None
        self.__createFitDatasets()
        """
        read first data + guess chunks
        while chunks are not empty:
            call optimize on this data
            write the fit to the H5 dataset
            request for next chunk
        """
        pass

    # def __optimize(self, func, data, guess, solver, parallel='multiprocess', processors=min(1,abs(mp.cpu_count()-2)), **kwargs):
        """
        Parameters:
        -----
        func : callable
            Function of the parameters.
        data : nd array
            Main data chunk
        guess: nd array
            Initial guess for this data chunk
        solver : string
            Optimization solver to use (minimize,least_sq, etc...). For additional info see scipy.optimize
        parallel : string
            Type of distributed computing to use. Currently, only 'multiprocess' (a variant of multiprocessing
            uses dill instead of pickle) is implemented. But Spark and MPI will be implemented in the future.
        processors : int, optional
            Number of processors to use. Default is all of them - 2 .
        **kwargs:
            Additional keyword arguments that are passed on to the solver.

        Returns:
        -------
        Results of the optimization.

        """
        # try:
        #     self.solver = scipy.optimize.__dict__[solver]
        # except KeyError:
        #     warn('Solver %s does not exist!' %(solver))
        #
        # def _callSolver(data, guessChunk):
        #     results = self.solver.__call__(func,guessChunk,**kwargs)
        #     return results
        #
        # if parallel=='multiprocess':
        #     # start pool of workers
        #     print('launching %i kernels...'%(processors))
        #     pool = mp.Pool(processors)
        #     # Divvy up the tasks and run them
        #     tasks = [(dataChunk,guessChunk) for (dataChunk,guessChunk) in zip(data,guess)]
        #     chunk = int(data.shape[0]/processors)
        #     jobs = pool.imap(_callSolver, tasks, chunksize = chunk)
        #     # Collect the results
        #     results =[]
        #     print('Extracting Peaks...')
        #     try:
        #         for j in jobs:
        #             results.append(j)
        #     except ValueError:
        #         warn('It appears that one of the jobs failed.')