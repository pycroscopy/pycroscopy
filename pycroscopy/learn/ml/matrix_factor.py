# Matrix Factorization class
# Instantiate this object and perform matrix factorization on any sidpy dataset
# Note that matrix factorization can only operate on 2D matrices. So we will automatically reshape
# all datasets to be two dimensional (spatial v spectral)
import sidpy
import numpy as np
import matplotlib.pyplot as plt

class MatrixFactor():
    def __init__(self, data, method = 'svd', n_components = 5, normalize = False) -> None:
        """
        Parameters:
            - data: (sidpy.Dataset)
            - 
        """
        self.data = data
        self.allowed_methods = ['svd', 'nmf', 'ica', 'nfindr', 'kernelpca'] #Might want to add others...
        assert method in self._allowed_methods, "Method must be one of {}".format(self._allowed_methods)
        self.method = method
        self.normalize = normalize
        self.data_2d = _return_2d_dataset(self, self.data)
        self.results_computed = False

    def _return_2d_dataset(self, data):
        # here goes the code for flattening the dataset into a 2D block
        # Mani's code will be used here. So ignore this for now.
        return


    def do_fit(self) -> sidpy.Dataset.dataset:
        """
        Parameters:
        (none)

        Returns: 
        Sidpy dataset after fit operation. Fit will calculate according to the method chosen.

        """
        
        if self.method == 'svd':
            u,s,v = np.linalg.svd(self.data_2d)
            components = s*v.T #check...
            abundances = u
            #components = ...
            #abundances = ...
        elif self.method =='nmf':
            # code goes here...
            nmf = NMF(n_components = self.num_components)
            nmf.fit_transform(self.data_2d[:])
            components = nmf.components_
            

        #Then we want to return components, abundances, and explained variances if available.
        # We should return a list of sidpy dataset objects
        self.components = components
        self.abundances = abundances
        self.results_computed  = True

        return components, abundances

    def plot_results(self):
        if self.results_computed is False:
            raise RuntimeError("No results are available. Call 'do_fit()' method first")
        self._plot_abundances(self.abundances)
        self._plot_components(self.components)

    def _plot_abundances(self, abundances)-> plt.figure:
        # We may need to think about this a little it
        # Note that we have some viz in sidpy already. We can try to leverage.
        fig, axes = plt.subplots()

        return fig

    def _plot_components(self, abundances)-> plt.figure:
        # We may need to think about this a little it
        fig, axes = plt.subplots()
        
        return fig
