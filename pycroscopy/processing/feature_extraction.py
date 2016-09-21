# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:34:12 2015

@author: Numan Laanait -- nlaanait@gmail.com
"""

import warnings

import h5py
import numpy as np
import skimage.feature


#TODO: Docstrings following numpy standard.

#### Functions
def pickle_keypoints(keypoints):
    ''' Function to pickle cv2.sift keypoint objects
    '''
    kpArray = np.array([])
    for point in keypoints:
        kpArray =np.append(kpArray, [point.pt[1],point.pt[0]])
    kpArray =np.reshape(kpArray, (int(kpArray.size/2), 2))
    return kpArray



# Class to do feature extraction. This is a wrapper on scikit-image and openCV feature extraction detectors.
#TODO: Add support for opencV or implement sift.
#TODO: Add io operations for extracted features.
#TODO: Memory checking, since some of the features are quite large.

class FeatureExtractorParallel(object):
    ''' This is an Object used to contain a data set and has methods to perform
        feature extraction on the data set that are detector based.
        Begin by loading a detector for features and a computer vision library.
        Input:
            name: (string) name of detector.
            lib: (string) computer vision library to use (opencv or skimage)
            The following can be used for:
            lib = opencv:
                SIFT
                ORB
                SURF
                ...
            lib = skimage:
                ORB
                BRIEF
                CENSURE
                ...
    '''
    def __init__(self, detector_name, lib):
        self.data = []
        self.lib = lib

        try:
            if self.lib == 'opencv':
                pass
    #                detector = cv2.__getattribute__(detector_name)
            elif self.lib == 'skimage':
                self.detector = skimage.feature.__getattribute__(detector_name)
        except AttributeError:
            print('Error: The Library does not contain the specified detector')

    def clearData(self):
        del self.data
        self.data = []

    def loadData(self, dataset):
        ''' This is a Method that loads h5 Dataset to be corrected.
            input: h5 dataset
        '''
        if not isinstance(dataset, h5py.Dataset):
            warnings.warn('Error: Data must be an h5 Dataset object')
        else:
            self.data = dataset
            dim = int(np.sqrt(self.data.shape[-1]))
            self.data = self.data.reshape(-1,dim,dim)

    def getData(self):
        ''' This is a Method that returns the loaded h5 Dataset.
            output: h5 dataset
        '''
        return self.data

    def getFeatures(self, **kwargs):
        ''' This is a Method that returns features (keypoints and descriptors)
            that are obtained by using the FeatureExtractor.Detector object.
            input:
                processors: int, optional
                            Number of processors to use, default = 1.
                mask: boolean, optional, default False.
                    Whether to use
            output: keypoints, descriptors
        '''
        detector = self.detector
        dset = self.data
        lib = self.lib
        processes = kwargs.get('processors', 1)
        mask = kwargs.get('mask', False)
        origin = kwargs.get('origin',[0,0])
        winSize= kwargs.get('window_size', 0)

        if mask:
            def mask_func(x, winSize):
                x[origin[0]- winSize/2: origin[0]+ winSize/2, origin[1]- winSize/2: origin[1]+ winSize/2] = 2
                x = x - 1
                return x
            mask_ind = np.mask_indices(dset.shape[-1], mask_func, winSize)
            self.data = np.array([ imp[mask_ind].reshape(winSize, winSize) for imp in dset])

        # detect and compute keypoints
        def detect(image):
            if lib =='opencv':
                image = (image - image.mean())/image.std()
                image = image.astype('uint8')
                k_obj, d_obj = detector.detectAndCompute(image, None)
                keypts, descs = pickle_keypoints(k_obj), pickle_keypoints(d_obj)

            elif lib == 'skimage':
                imp = (image-image.mean())/np.std(image)
                imp[imp < 0] = 0
                imp.astype('float32')
                detector.detect_and_extract(imp)
                keypts, descs = detector.keypoints, detector.descriptors

            return keypts, descs

        # start pool of workers
        print('launching %i kernels...'%(processes))
        pool = multiProcess.Pool(processes)
        tasks = [(imp) for imp in self.data]
        chunk = int(self.data.shape[0]/processes)
        jobs = pool.imap(detect, tasks, chunksize = chunk)

        # get keypoints and descriptors
        results =[]
        print('Extracting features...')
        try:
            for j in jobs:
                results.append(j)
        except ValueError:
            warnings.warn('ValueError something about 2d-image. Probably some of the detector input params are wrong.')

        keypts = [itm[0].astype('int') for itm in results]
        desc = [itm[1] for itm in results]

        # close the pool
        print('Closing down the kernels... \n')
        pool.close()

        return keypts, desc

class FeatureExtractorSerial(object):
    ''' This is an Object used to contain a data set and has methods to perform
        feature extraction on the data set that are detector based.
        Begin by loading a detector for features and a computer vision library.
        Input:
            name: (string) name of detector.
            lib: (string) computer vision library to use (opencv or skimage)
            The following can be used for:
            lib = opencv:
                SIFT
                ORB
                SURF
                ...
            lib = skimage:
                ORB
                BRIEF
                CENSURE
                ...
    '''
    def __init__(self, detector_name, lib):
        self.data = []
        self.lib = lib

        try:
            if self.lib == 'opencv':
                pass
    #                detector = cv2.__getattribute__(detector_name)
            elif self.lib == 'skimage':
                self.detector = skimage.feature.__getattribute__(detector_name)
        except AttributeError:
            print('Error: The Library does not contain the specified detector')

    def clearData(self):
        del self.data
        self.data = []

    def loadData(self, dataset):
        ''' This is a Method that loads h5 Dataset to be corrected.
            input: h5 dataset
        '''
        if not isinstance(dataset, h5py.Dataset):
            warnings.warn('Error: Data must be an h5 Dataset object')
        else:
            self.data = dataset
            dim = int(np.sqrt(self.data.shape[-1]))
            self.data = self.data.reshape(-1,dim,dim)

    def getData(self):
        ''' This is a Method that returns the loaded h5 Dataset.
            output: h5 dataset
        '''
        return self.data

    def getFeatures(self, **kwargs):
        ''' This is a Method that returns features (keypoints and descriptors)
            that are obtained by using the FeatureExtractor.Detector object.
            input:
                mask: boolean, optional, default False.
                    Whether to use
            output: keypoints, descriptors
        '''
        detector = self.detector
        dset = self.data
        lib = self.lib
        mask = kwargs.get('mask', False)
        origin = kwargs.get('origin',[0,0])
        winSize= kwargs.get('window_size', 0)

        if mask:
            def mask_func(x, winSize):
                x[origin[0]- winSize/2: origin[0]+ winSize/2, origin[1]- winSize/2: origin[1]+ winSize/2] = 2
                x = x - 1
                return x
            mask_ind = np.mask_indices(dset.shape[-1], mask_func, winSize)
            self.data = np.array([ imp[mask_ind].reshape(winSize, winSize) for imp in dset])

        # detect and compute keypoints
        def detect(image):
            if lib =='opencv':
                image = (image - image.mean())/image.std()
                image = image.astype('uint8')
                k_obj, d_obj = detector.detectAndCompute(image, None)
                keypts, descs = pickle_keypoints(k_obj), pickle_keypoints(d_obj)

            elif lib == 'skimage':
                imp = (image-image.mean())/np.std(image)
                imp[imp < 0] = 0
                imp.astype('float32')
                detector.detect_and_extract(imp)
                keypts, descs = detector.keypoints, detector.descriptors

            return keypts, descs

        # start pool of workers
        results = [detect(imp) for imp in self.data]

        # get keypoints and descriptors
        keypts = [itm[0].astype('int') for itm in results]
        desc = [itm[1] for itm in results]

        return keypts, desc