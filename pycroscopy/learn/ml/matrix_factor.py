# Matrix Factorization class
# Instantiate this object and perform matrix factorization on any sidpy dataset
# Note that matrix factorization can only operate on 2D matrices. So we will automatically reshape
# all datasets to be two dimensional (spatial v spectral)
import sidpy
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from matplotlib import gridspec

class MatrixFactor():
    def __init__(self, data, method = 'svd', n_components = 5, normalize = False) -> None:
        """
        Parameters:
            - data: (sidpy.Dataset)
            - None
        """
        self.data = data
        self.shape = data.shape()
        self.data_type = data.data_type
        self.allowed_methods = ['svd', 'nmf', 'ica', 'nfindr', 'kernelpca'] #Might want to add others...
        assert method in self._allowed_methods, "Method must be one of {}".format(self._allowed_methods)
        self.method = method
        self.normalize = normalize
        self.data_2d = self._return_2d_dataset(self, self.data)
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
            print("Performing SVD fitting")
            svd = decomposition.TruncatedSVD(n_components=self.n_components)
            abundances = svd.fit_transform(self.data_2d)
            components = svd.components_
         
        elif self.method =='nmf':
            print("Performing NMF fitting")
            nmf = decomposition.NMF(n_components = self.n_components)
            abundances = nmf.fit_transform(self.data_2d)
            components = nmf.components_

        elif self.method == 'pca':
            print("Performing PCA fitting")
            pca = decomposition.PCA(n_components=self.n_components)
            abundances = pca.fit_transform(self.data_2d)
            components = pca.components_

        elif self.method == 'ica':
            print("Performing ICA fitting")
            ica = decomposition.FastICA(n_components=self.n_components)
            abundances = ica.fit_transform(self.data_2d)
            components = ica.components_
            

        #return components and abundances.
        self.components = components
        self.abundances = abundances
        self.results_computed  = True

        print("Fitting completed, run plot_results to look at the results")

        return components, abundances

    def plot_results(self):
        if self.results_computed is False:
            raise RuntimeError("No results are available. Call 'do_fit()' method first")
        self._plot_abundances(self.abundances)
        self._plot_components(self.components)

    def _plot_abundances(self, abundances)-> plt.figure:
        print('Abundances')
        gs = gridspec.GridSpec(int(self.n_components//3)+1, 3)
        fig = plt.figure(figsize = (4*(int(self.n_components//3)+1), 4*(4//1.5)))
        if len(self.shape) == 2:
            for i in range(self.n_components):
                ax = fig.add_subplot(gs[i])
                ax.plot(self.abundances[:, i])
                ax.set_title('Component ' + str(i + 1))
            plt.show()
        elif self.data_type == "SPECTRAL_IMAGE":
            d1, d2, _ = self.shape
            for i in range(self.n_components):
                ax = fig.add_subplot(gs[i])
                ax.imshow(self.abundances[:, i].reshape(d1, d2))
                ax.set_title('Component ' + str(i + 1))
            plt.show()
        elif self.data_type == "IMAGE_STACK":
            for i in range(self.n_components):
                ax = fig.add_subplot(gs[i])
                ax.plot(self.abundances[:, i])
                ax.set_title('Component ' + str(i + 1))
            plt.show()

        return fig

    def _plot_components(self, abundances)-> plt.figure:
        print('Components')

        gs = gridspec.GridSpec(int(self.n_components//3)+1, 3)
        fig = plt.figure(figsize = (4*(int(self.n_components//3)+1), 4*(4//1.5))))
        if len(self.shape) == 2:   
            for i in range(self.n_components):
                ax = fig.add_subplot(gs[i])
                ax.plot(components[i])
                ax.set_title('Component ' + str(i + 1))
            plt.show()
        elif self.data_type == "SPECTRAL_IMAGE":
            for i in range(self.n_components):
                ax = fig.add_subplot(gs[i])
                ax.plot(components[i])
                ax.set_title('Component ' + str(i + 1))
            plt.show()
        elif self.data_type == "IMAGE_STACK":
            d1, d2, _ = self.shape 
            for i in range(self.n_components):
                ax = fig.add_subplot(gs[i])
                ax.plot(components[i].reshape(d1, d2))
                ax.set_title('Component ' + str(i + 1))
            plt.show()
        
        return fig
