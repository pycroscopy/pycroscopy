"""
Created on 12/15/16 10:08 AM
@author: Numan Laanait
"""

from __future__ import division, print_function, absolute_import, unicode_literals
from warnings import warn
import numpy as np
import sys
import joblib
from .guess_methods import GuessMethods
from .fit_methods import Fit_Methods
import scipy


def targetFuncGuess(args, **kwargs):
    """
    This is just creates mappable function for multiprocessing guess
    :param args:
    :param kwargs:
    :return:
    """
    opt = args[-1]
    func = opt._guessFunc()
    results = func(args[0])
    return results


def targetFuncFit(args, **kwargs):
    """
    Needed to create mappable function for multiprocessing
    :param args:
    :param kwargs:
    :return:
    """
    opt = args[-1]
    solver, solver_options, func = opt._initiateSolverAndObjFunc()
    results = solver(func, args[1], args=[args[0]])
    return results


class Optimize(object):
    """
    In charge of all optimization and computation and is used within the Model Class.
    """

    def __init__(self, data=np.array([]), guess=np.array([]), parallel=True):
        """

        :param data:
        """
        if isinstance(data, np.ndarray):
            self.data = data
        if isinstance(guess, np.ndarray):
            self.guess = guess
        else:
            warn('Error: data and guess must be numpy.ndarray. Exiting...')
            sys.exit()
        self._parallel = parallel
        self.strategy = None
        self.options = None
        self.solver_type = None
        self.solver_options = None

    def _guessFunc(self):
        gm = GuessMethods()
        if self.strategy in gm.methods:
            func = gm.__getattribute__(self.strategy)(**self.options)
            return func
        else:
            warn('Error: %s is not implemented in pycroscopy.analysis.GuessMethods to find guesses' % self.strategy)

    def computeGuess(self, processors=1, strategy='wavelet_peaks',
                     options={"peak_widths": np.array([10, 200]), "peak_step": 20}, **kwargs):
        """
        Computes the guess function using numerous cores

        Parameters
        ----------
        processors : unsigned int
            Number of logical cores to use for computing
        strategy: string
            Default is 'Wavelet_Peaks'.
            Can be one of ['wavelet_peaks', 'relative_maximum', 'gaussian_processes']. For updated list,
            run GuessMethods.methods
        options: dict
            Default: Options for wavelet_peaks{"peaks_widths": np.array([10,200]), "peak_step":20}.
            Dictionary of options passed to strategy. For more info see GuessMethods documentation.

        kwargs:
            processors: int
                number of processors to use. Default all processors on the system except for 1.

        Returns
        -------
        results : unknown
            unknown
        """
        self.strategy = strategy
        self.options = options
        gm = GuessMethods()
        if strategy in gm.methods:
            func = gm.__getattribute__(strategy)  # (**options)
            # start pool of workers
            if processors > 1:
                print('Computing Jobs In parallel ... launching %i kernels...' % processors)
            else:
                print("Computing Guesses In Serial ...")

            values = [joblib.delayed(func)(x, **options) for x in self.data]
            results = joblib.Parallel(n_jobs=processors)(values)

            return results
        else:
            warn('Error: %s is not implemented in pycroscopy.analysis.GuessMethods to find guesses' % strategy)

    def _initiateSolverAndObjFunc(self, obj_func):
        fm = Fit_Methods()

        if obj_func['class'] is None:
            self.obj_func = obj_func['obj_func']
        else:
            self.obj_func_name = obj_func.pop('obj_func')
            self.obj_func = fm.__getattribute__(self.obj_func_name)
            self.obj_func_class = obj_func.pop('class')
            self.obj_func_args = obj_func.values()

    def computeFit(self, processors=1, solver_type='least_squares', solver_options={},
                   obj_func={'class': 'Fit_Methods', 'obj_func': 'SHO', 'xvals': np.array([])}):
        """

        Parameters
        ----------
        processors : unsigned int
            Number of logical cores to use for computing
        solver_type : string
            Optimization solver to use (minimize,least_sq, etc...). For additional info see scipy.optimize
        solver_options: dict()
            Default: dict()
            Dictionary of options passed to solver. For additional info see scipy.optimize
        obj_func: dict()
            Default is 'SHO'.
            Can be one of ['wavelet_peaks', 'relative_maximum', 'gaussian_processes'].
            For updated list, run GuessMethods.methods

        Returns
        -------
        results : unknown
            unknown
        """
        self.solver_type = solver_type
        self.solver_options = solver_options
        if self.solver_type not in scipy.optimize.__dict__.keys():
            warn('Solver %s does not exist!. For additional info see scipy.optimize' % solver_type)
            sys.exit()

        self._initiateSolverAndObjFunc(obj_func)

        if processors > 1:
            print('Computing Jobs In parallel ... launching %i kernels...' % processors)
        else:
            print("Computing Guesses In Serial ...")

        solver = scipy.optimize.__dict__[self.solver_type]
        values = [joblib.delayed(solver)(self.obj_func, guess,
                                         args=[vector] + list(self.obj_func_args),
                                         **solver_options) for vector, guess in zip(self.data, self.guess)]
        results = joblib.Parallel(n_jobs=processors)(values)

        return results

        # if self._parallel:
        #     # start pool of workers
        #     print('Computing Jobs In parallel ... launching %i kernels...' % processors)
        #     pool = mp.Pool(processors)
        #     # Vectorize tasks
        #     tasks = [(vector, guess, self) for vector, guess in zip(self.data, self.guess)]
        #     chunk = int(self.data.shape[0] / processors)
        #     # Map them across processors
        #     jobs = pool.imap(targetFuncFit, tasks, chunksize=chunk)
        #     # get Results from different processes
        #     results = [j for j in jobs]
        #     # Finished reading the entire data set
        #     print('closing %i kernels...' % processors)
        #     pool.close()
        #     return results
        #
        # else:
        #     print("Computing Fits In Serial ...")
        #     tasks = [(vector, guess, self) for vector, guess in zip(self.data, self.guess)]
        #     results = [targetFuncFit(task) for task in tasks]
        #     return results
