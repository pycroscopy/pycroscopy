"""
Created on 7/17/16 10:08 AM
@author: Numan Laanait, Suhas Somnath
"""

from warnings import warn
import numpy as np
import sys
import multiprocessing as mp
from .guess_methods import GuessMethods


def targetFunc(args,**kwargs):
    """
    Needed to create mappable function for multiprocessing
    :param args:
    :param kwargs:
    :return:
    """
    func = Optimize._guessFunc(args[-1])
    results = func(args[0])
    return results

class Optimize(object):
    """
    In charge of all optimization and computation and is used within the Model Class.
    """

    def __init__(self, data=np.array([]), parallel=True):
        """

        :param data:
        """
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            warn('Error: data must be numpy.ndarray. Exiting...')
            sys.exit()
        self._parallel = parallel


    def _guessFunc(self):
        gm = GuessMethods()
        if self.strategy in gm.methods:
            func = gm.__getattribute__(self.strategy)(**self.options)
            return func
        else:
            warn('Error: %s is not implemented in pycroscopy.analysis.GuessMethods to find guesses' % strategy)



    def computeGuess(self, processors = 1, strategy='wavelet_peaks',
                     options={"peak_widths": np.array([10,200]),"peak_step":20}, **kwargs):
        """

        Parameters
        ----------
        data
        strategy: string
            Default is 'Wavelet_Peaks'.
            Can be one of ['wavelet_peaks', 'relative_maximum', 'gaussian_processes']. For updated list, run GuessMethods.methods
        options: dict
            Default: Options for wavelet_peaks{"peaks_widths": np.array([10,200]), "peak_step":20}.
            Dictionary of options passed to strategy. For more info see GuessMethods documentation.

        kwargs:
            processors: int
                number of processors to use. Default all processors on the system except for 1.

        Returns
        -------

        """
        self.strategy = strategy
        self.options = options
        processors = processors
        gm = GuessMethods()
        if strategy in gm.methods:
            # func = gm.__getattribute__(strategy)(**options)
            results = list()
            if self._parallel:
                # start pool of workers
                print('Computing Jobs In parallel ... launching %i kernels...' % processors)
                pool = mp.Pool(processors)
                # Vectorize tasks
                tasks = [(vector,self) for vector in self.data]
                chunk = int(self.data.shape[0] / processors)
                # Map them across processors
                jobs = pool.imap(targetFunc, tasks, chunksize=chunk)
                # get Results from different processes
                results = [j for j in jobs]
                print('Extracted Results...')
                return results
                # Finished reading the entire data set
                print('closing %i kernels...' % processors)
                pool.close()
            else:
                print("Computing Guesses In Serial ...")
                results = [self._guessFunc(vector) for vector in self.data]
                return results
        else:
            warn('Error: %s is not implemented in pycroscopy.analysis.GuessMethods to find guesses' % strategy)
