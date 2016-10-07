"""
Created on Thu Oct  8 10:40:43 2015
@author: Numan Laanait -- nlaanait@gmail.com
"""

import math
import warnings

import h5py
import numpy as np
from skimage.feature import match_descriptors, register_translation
from skimage.measure import ransac
from skimage.transform import warp, SimilarityTransform


#TODO: Docstrings following numpy standard.

# Functions
def euclidMatch(Matches, keypts1, keypts2, misalign):
    """ Function that thresholds the matches, found from a comparison of
    their descriptors, by the maximum expected misalignment.
    """
    filteredMatches = np.array([])
    deltaX =(keypts1[Matches[:,0],:][:,0]-keypts2[Matches[:,1],:][:,0])**2
    deltaY =(keypts1[Matches[:,0],:][:,1]-keypts2[Matches[:,1],:][:,1])**2
    dist = np.apply_along_axis(np.sqrt, 0, deltaX + deltaY)
    filteredMatches = np.where(dist[:] < misalign, True, False)
    return filteredMatches


# function is taken as is from scikit-image.
def _center_and_normalize_points(points):
    """Center and normalize image points.

    The points are transformed in a two-step procedure that is expressed
    as a transformation matrix. The matrix of the resulting points is usually
    better conditioned than the matrix of the original points.

    Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    Parameters
    ----------
    points : (N, 2) array
        The coordinates of the image points.

    Returns
    -------
    matrix : (3, 3) array
        The transformation matrix to obtain the new points.
    new_points : (N, 2) array
        The transformed image points.

    """

    centroid = np.mean(points, axis=0)

    rms = math.sqrt(np.sum((points - centroid) ** 2) / points.shape[0])

    norm_factor = math.sqrt(2) / rms

    matrix = np.array([[norm_factor, 0, -norm_factor * centroid[0]],
                       [0, norm_factor, -norm_factor * centroid[1]],
                       [0, 0, 1]])

    pointsh = np.row_stack([points.T, np.ones((points.shape[0]),)])

    new_pointsh = np.dot(matrix, pointsh).T

    new_points = new_pointsh[:, :2]
    new_points[:, 0] /= new_pointsh[:, 2]
    new_points[:, 1] /= new_pointsh[:, 2]

    return matrix, new_points





class TranslationTransform(object):
    """ 2D translation using homogeneous representation:

    The transformation matrix is:
        [[1  1  tX]
         [1  1  tY]
         [0  0  1]]
         X: translation of x-axis.
         Y: translation of y-axis.

    Parameters:

    translation: (tX, tY) as a tuple.

    Attributes:

    params : (3, 3) array
        Homogeneous transformation matrix.

    """

    def __init__(self, matrix = None, translation = None):
        params = translation

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            if matrix.shape != (3, 3):
                raise ValueError("Invalid shape of transformation matrix.")
            self.params = matrix

        elif params:
            if translation is None:
                translation = (0., 0.)

            self.params = np.array([
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]
                ], dtype = 'float32')
            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(3)

    def estimate(self, src, dst):
     #evaluate transformation matrix from src, dst
     # coordinates
        try:
            xs = src[:, 0][0]
            ys = src[:, 1][1]
            xd = dst[:, 0][0]
            yd = dst[:, 1][1]
            S = np.array([[1., 0., xd-xs],
                          [0., 1., yd-ys],
                          [0., 0., 1.]
                          ],dtype = 'float32')
            self.params = S
            return True
        except IndexError:
            return False

    @property
    def _inv_matrix(self):
        inv_matrix = self.params
        inv_matrix[0:2,2] = - inv_matrix[0:2,2]
        return inv_matrix

    def _apply_mat(self, coords, matrix):
        coords = np.array(coords, copy=False, ndmin=2)

        x, y = np.transpose(coords)
        src = np.vstack((x, y, np.ones_like(x)))
        dst = np.dot(src.transpose(), matrix.transpose())

        # rescale to homogeneous coordinates
        dst[:, 0] /= dst[:, 2]
        dst[:, 1] /= dst[:, 2]

        return dst[:, :2]

    def __call__(self, coords):
        return self._apply_mat(coords, self.params)

    def inverse(self, coords):
        """ Apply inverse transformation.

        Parameters
        ----------
        coords : (N, 2) array
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Transformed coordinates.

        """
        return self._apply_mat(coords, self._inv_matrix)
    def residuals(self, src, dst):
        """Determine residuals of transformed destination coordinates.

        For each transformed source coordinate the euclidean distance to the
        respective destination coordinate is determined.

        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

        Returns
        -------
        residuals : (N, ) array
            Residual for coordinate.

        """

        return np.sqrt(np.sum((self(src) - dst)**2, axis=1))

    @property
    def translation(self):
        return self.params[0:2, 2]





class RigidTransform(object):
    """ 2D translation using homogeneous representation:

    The transformation matrix is:
        [[cos(theta)  -sin(theta)  tX]
         [sin(theta)  cos(theta)   tY]
         [0             0           1]]
         X: translation along x-axis.
         Y: translation along y-axis.
         theta: rotation angle in radians.

    Parameters:

    translation: (tX, tY) as a tuple.
    rotation: float in radians.

    Attributes:

    params : (3, 3) array
        Homogeneous transformation matrix.

    """

    def __init__(self, matrix = None, rotation = None, translation = None):
        params = any(param is not None
                     for param in (rotation, translation))

        if params and matrix is not None:
            raise ValueError("You cannot specify the transformation matrix and"
                             " the implicit parameters at the same time.")
        elif matrix is not None:
            if matrix.shape != (3, 3):
                raise ValueError("Invalid shape of transformation matrix.")
            self.params = matrix

        elif params:
            if translation is None:
                translation = (0, 0)
            if rotation is None:
                rotation = 0

            self.params = np.array([
                [math.cos(rotation), - math.sin(rotation), 0],
                [math.sin(rotation),   math.cos(rotation), 0],
                [                 0,                    0, 1]
            ])

            self.params[0:2, 2] = translation
        else:
            # default to an identity transform
            self.params = np.eye(3)

    def estimate(self, src, dst):
        """Set the transformation matrix with the explicit parameters.

        You can determine the over-, well- and under-determined parameters
        with the total least-squares method.

        Number of source and destination coordinates must match.

        The transformation is defined as::

            X = a0 * x - b0 * y + a1
            Y = b0 * x + a0 * y + b1

        These equations can be transformed to the following form::

            0 = a0 * x - b0 * y + a1 - X
            0 = b0 * x + a0 * y + b1 - Y

        which exist for each set of corresponding points, so we have a set of
        N * 2 equations. The coefficients appear linearly so we can write
        A x = 0, where::

            A   = [[x 1 -y 0 -X]
                   [y 0  x 1 -Y]
                    ...
                    ...
                  ]
            x.T = [a0 a1 b0 b1 c3]

        In case of total least-squares the solution of this homogeneous system
        of equations is the right singular vector of A which corresponds to the
        smallest singular value normed by the coefficient c3.

        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """

        try:
            src_matrix, src = _center_and_normalize_points(src)
            dst_matrix, dst = _center_and_normalize_points(dst)
        except ZeroDivisionError:
            self.params = np.nan * np.empty((3, 3))
            return False

        xs = src[:, 0]
        ys = src[:, 1]
        xd = dst[:, 0]
        yd = dst[:, 1]
        rows = src.shape[0]

        # params: a0, a1, b0, b1
        A = np.zeros((rows * 2, 5))
        A[:rows, 0] = xs
        A[:rows, 2] = - ys
        A[:rows, 1] = 1
        A[rows:, 2] = xs
        A[rows:, 0] = ys
        A[rows:, 3] = 1
        A[:rows, 4] = xd
        A[rows:, 4] = yd

        _, _, V = np.linalg.svd(A)

        # solution is right singular vector that corresponds to smallest
        # singular value
        a0, a1, b0, b1 = - V[-1, :-1] / V[-1, -1]

        S = np.array([[a0, -b0, a1],
                      [b0,  a0, b1],
                      [ 0,   0,  1]])

        # De-center and de-normalize
        S = np.dot(np.linalg.inv(dst_matrix), np.dot(S, src_matrix))

        self.params = S

        return True


    def _apply_mat(self, coords, matrix):
        coords = np.array(coords, copy=False, ndmin=2)

        x, y = np.transpose(coords)
        src = np.vstack((x, y, np.ones_like(x)))
        dst = np.dot(src.transpose(), matrix.transpose())

        # rescale to homogeneous coordinates
        dst[:, 0] /= dst[:, 2]
        dst[:, 1] /= dst[:, 2]

        return dst[:, :2]

    def __call__(self, coords):
        return self._apply_mat(coords, self.params)

    def inverse(self, coords):
        ''' Apply inverse transformation.

        Parameters
        ----------
        coords : (N, 2) array
            Source coordinates.

        Returns
        -------
        coords : (N, 2) array
            Transformed coordinates.

        '''
        return self._apply_mat(coords, self._inv_matrix)
    def residuals(self, src, dst):
        """Determine residuals of transformed destination coordinates.

        For each transformed source coordinate the euclidean distance to the
        respective destination coordinate is determined.

        Parameters
        ----------
        src : (N, 2) array
            Source coordinates.
        dst : (N, 2) array
            Destination coordinates.

        Returns
        -------
        residuals : (N, ) array
            Residual for coordinate.

        """

        return np.sqrt(np.sum((self(src) - dst)**2, axis=1))


    @property
    def _inv_matrix(self):
        return np.linalg.inv(self.params)

    @property
    def rotation(self):
        return math.atan2(self.params[1, 0], self.params[1, 1])

    @property
    def translation(self):
        return self.params[0:2, 2]



# Class to do geometric transformations. This is a wrapper on scikit-image functionality.
# TODO: io operations for features and optical geometric transformations.

class geoTransformerParallel(object):
    ''' This object contains methods to perform geometric transformations on
    a sequence of images. Some of the capabilities are:
    + Homography by feature extraction.
    + Intensity-based image registration.
    + Projection Correction.
    '''

    def __init__(self):
        self.__init__
        self.data = []
        self.features = []

    def clearData(self):
        ''' This is a Method to clear the data from the object.
        '''
        del self.data
        self.data = []

    def loadData(self, dataset):
        ''' This is a Method that loads h5 Dataset to be corrected.
            input: h5 dataset
        '''
        if not isinstance(dataset, h5py.Dataset):
            warnings.warn( 'Error: Data must be an h5 Dataset object'   )
        else:
            self.data = dataset
            dim = int(np.sqrt(self.data.shape[-1]))
            self.data = self.data.reshape(-1,dim,dim)

    def loadFeatures(self, features):
        ''' This is a Method that loads features to be used for homography etc ...
        input:
            features : [keypoints, descriptors].
                These can come from FeatureExtractor.getFeatures() or elsewhere.
                The format is :
                    keypoints = [np.ndarray([y_position, x_position])]
                    descriptors = [np.ndarray()]
        '''
        self.features = features

    def matchFeatures(self, **kwargs):
        ''' This is a Method that computes similarity between keypoints based on their
        descriptors. Currently only skimage.feature.match_descriptors is implemented.
        In the future will need to add opencv2.matchers.
        Input:
            processors: int, optional
                    Number of processors to use, default = 1.
            maximum_distance: int, optional
                    maximum_distance (int) of misalignment, default = infinity.
                    Used to filter the matches before optimizing the transformation.
        Output:
            Matches.
        '''
        desc = self.features[-1]
        keypts = self.features[0]
        processes = kwargs.get('processors', 1)
        maxDis = kwargs.get('maximum_distance', np.infty)


        def match(desc):
            desc1, desc2 = desc[0], desc[1]
            matches = match_descriptors(desc1, desc2, cross_check=True)
            return matches

        # start pool of workers
        pool = multiprocess.Pool(processes)
        print('launching %i kernels...'%(processes))

        tasks = [ (desc1, desc2) for desc1, desc2 in zip(desc[:],desc[1:]) ]
        chunk = int(len(desc)/processes)
        jobs = pool.imap(match, tasks, chunksize = chunk)

        # get matches
        print('Extracting Matches From the Descriptors...')

        matches =[]
        for j in jobs:
            matches.append(j)

        # close the pool
        print('Closing down the kernels...\n')
        pool.close()

        # impose maximum_distance misalignment constraints on matches
        filt_matches = []
        for match, key1, key2 in zip(matches, keypts[:],keypts[1:]):
            filteredMask = euclidMatch(match, key1, key2, maxDis)
            filt_matches.append(match[filteredMask])


        return matches, filt_matches


    def findTransformation(self, transform, matches, processes, **kwargs):
        ''' This is a Method that finds the optimal transformation between two images
        given matching features using a random sample consensus.
            Input:
                transform: skimage.transform object
                matches (list): matches found through match_features method.
                processors: Number of processors to use.
                **kwargs are passed to skimage.transform.ransac

            Output:
                Transformations.
        '''

        keypts = self.features[0]

        def optimization(Pts):
            robustTrans, inliers = ransac((Pts[0], Pts[1]), transform, **kwargs)
            output = [robustTrans, inliers]
            return output

         # start pool of workers
        print('launching %i kernels...'%(processes))
        pool = mp.Pool(processes)
        tasks = [ (key1[match[:, 0]], key2[match[:, 1]])
                    for match, key1, key2 in zip(matches,keypts[:],keypts[1:]) ]
        chunk = int(len(keypts)/processes)
        jobs = pool.imap(optimization, tasks, chunksize = chunk)

        # get Transforms and inlier matches
        transforms, trueMatches =[], []
        print('Extracting Inlier Matches with RANSAC...')
        try:
            for j in jobs:
                transforms.append(j[0])
                trueMatches.append(j[1])
        except np.linalg.LinAlgError:
            pass

        # close the pool
        pool.close()
        print('Closing down the kernels...\n')

        return transforms, trueMatches


    #TODO: Need parallel version for transforming stack of images.
    def applyTransformation(self, transforms, **kwargs):
        ''' This is the method that takes the list of transformation found by findTransformation
         and applies them to the data set.

         Input:
             transforms: (list of skimage.GeoemetricTransform objects).
                     The objects must be inititated with the desired parameters.
             transformation: string, optional.
                     The type of geometric transformation to use (i.e. translation, rigid, etc..)
                     Currently, only translation is implemented.
                     default, translation.
             origin: int, optional
                     The position in the data to take as origin, i.e. don't transform.
                     default, center image in the stack.
             processors: int, optional
                    Number of processors to use, default = 1.
                    Currently,only one processor is used.

        Output:
            Transformed images, transformations

        '''
        dic = ['processors','origin','transformation']
        for key in kwargs.keys():
            if key not in dic:
                print('%s is not a parameter of this function' %(str(key)))

        processes = kwargs.get('processors', 1)
        origin = kwargs.get('origin', int(self.data.shape[0]/2))
        transformation = kwargs.get('transformation','translation')

        dset = self.data
        # For now restricting this to just translation... Straightforward to generalize to other transform objects.
        if transformation == 'translation':

            YTrans = np.array([trans.translation[0] for trans in transforms])
            XTrans = np.array([trans.translation[1] for trans in transforms])
            chainL = []
            for y, x in zip(range(0,YTrans.size+1), range(0,XTrans.size+1)):
                if y < origin:
                    ychain = -np.sum(YTrans[y:origin])
                    xchain = -np.sum(XTrans[x:origin])

                elif y > origin:
                    ychain = np.sum(YTrans[origin:y])
                    xchain = np.sum(XTrans[origin:x])
                else:
                    ychain = 0
                    xchain = 0

                chainL.append([xchain, ychain])

            chainTransforms = []
            for params in  chainL:
                T = TranslationTransform(translation = params)
                chainTransforms.append(T)

        # Just need a single function that does boths
        if transformation == 'rotation':

            rotTrans = np.array([trans.rotation for trans in transforms])
            YTrans = np.array([trans.translation[0] for trans in transforms])
            XTrans = np.array([trans.translation[1] for trans in transforms])
            chainL = []
            for x in range(0,rotTrans.size+1):
                if x < origin:
                    rotchain = -np.sum(rotTrans[x:origin])
                    ychain = -np.sum(YTrans[x:origin])
                    xchain = -np.sum(XTrans[x:origin])

                elif x > origin:
                    rotchain = np.sum(rotTrans[origin:x])
                    ychain = np.sum(YTrans[origin:x])
                    xchain = np.sum(XTrans[origin:x])
                else:
                    rotchain = 0
                    ychain = 0
                    xchain = 0

                chainL.append([rotchain, xchain, ychain])

            chainTransforms = []
            for params in  chainL:
                T = SimilarityTransform(scale = 1.0, rotation = np.deg2rad(params[0]), translation = (params[1],params[2]))
#                T = SimilarityTransform(rotation = params, translation = (0,0))
                chainTransforms.append(T)

        # Use the chain transformations to transform the dataset
        output_shape = dset[0].shape
#        output_shape = (2048, 2048)
        def warping(datum):
            imp, transform  = datum[0], datum[1]
            transimp = warp(imp, inverse_map= transform, output_shape = output_shape,
                            cval = 0, preserve_range = True)
            return transimp

#          #start pool of workers
#         #somehow wrap function crashes when run in parallel! run sequentially for now.
#        pool = mp.Pool(processes)
#        print('launching %i kernels...'%(processes))
#        tasks = [ (imp, transform) for imp, transform in zip(dset, chainTransforms) ]
#        chunk = int(dset.shape[0]/processes)
#        jobs = pool.imap(warping, tasks, chunksize = 1)
#        #close the pool
#        pool.close()
#        print('Closing down the kernels... \n')
#
        # get transformed images and pack into 3d np.ndarray
        print('Transforming Images...')
        transImages = np.copy(dset[:])

        for imp, transform, itm in zip( transImages, chainTransforms, range(0,transImages.shape[0])):
            transimp = warping([imp, transform])
            transImages[itm] = transimp
            print('Image #%i'%(itm))


        return transImages, chainTransforms

    def correlationTransformation(self, **kwargs):
        ''' Uses Cross-correlation to find a translation between 2 images.
            Input:
                Processors: int, optional
                    Number of processors to use, default = 1.

            Output:
                Transformations.
        '''

        processes = kwargs.get('processors', 1)

        pool = mp.Pool(processes)
        print('launching %i kernels...'%(processes))

        def register(images):
            imp1, imp2 = images[0], images[1]
            shifts, _, _ = register_translation(imp1,imp2)
            return shifts

        dim = int(np.sqrt(self.data.shape[-1]))
        tasks = [ (imp1, imp2)
                    for imp1, imp2 in zip(self.data[:], self.data[1:]) ]

        chunk = int((self.data.shape[0] - 1)/processes)
        jobs = pool.imap(register, tasks, chunksize = chunk)

        # get Transforms and inlier matches
        results = []
        print('Extracting Translations')
        try:
            for j in jobs:
                results.append(j)
        except:
            warnings.warn('Skipped Some Entry... dunno why!!')

        # close the pool
        pool.close()

        return results

class geoTransformerSerial(object):
    """ This object contains methods to perform geometric transformations on
    a sequence of images. Some of the capabilities are:
    + Homography by feature extraction.
    + Intensity-based image registration.
    + Projection Correction.
    """

    def __init__(self):
        self.__init__
        self.data = []
        self.features = []

    def clearData(self):
        """ This is a Method to clear the data from the object.
        """
        del self.data
        self.data = []

    def loadData(self, dataset):
        """ This is a Method that loads h5 Dataset to be corrected.
            input: h5 dataset
        """
        if not isinstance(dataset, h5py.Dataset):
            warnings.warn( 'Error: Data must be an h5 Dataset object'   )
        else:
            self.data = dataset
            dim = int(np.sqrt(self.data.shape[-1]))
            self.data = self.data.reshape(-1,dim,dim)

    def loadFeatures(self, features):
        """ This is a Method that loads features to be used for homography etc ...
        input:
            features : [keypoints, descriptors].
                These can come from FeatureExtractor.getFeatures() or elsewhere.
                The format is :
                    keypoints = [np.ndarray([y_position, x_position])]
                    descriptors = [np.ndarray()]
        """
        self.features = features

    def matchFeatures(self, **kwargs):
        """ This is a Method that computes similarity between keypoints based on their
        descriptors. Currently only skimage.feature.match_descriptors is implemented.
        In the future will need to add opencv2.matchers.
        Input:
            maximum_distance: int, optional
                    maximum_distance (int) of misalignment, default = infinity.
                    Used to filter the matches before optimizing the transformation.
        Output:
            Matches.
        """
        desc = self.features[-1]
        keypts = self.features[0]
        maxDis = kwargs.get('maximum_distance', np.infty)


        def match(desc):
            desc1, desc2 = desc[0], desc[1]
            matches = match_descriptors(desc1, desc2, cross_check=True)
            return matches

        # start pool of workers
        pool = mp.Pool(processes)
        print('launching %i kernels...'%(processes))

        tasks = [ (desc1, desc2) for desc1, desc2 in zip(desc[:],desc[1:]) ]
        chunk = int(len(desc)/processes)
        jobs = pool.imap(match, tasks, chunksize = chunk)

        # get matches
        print('Extracting Matches From the Descriptors...')

        matches =[]
        for j in jobs:
            matches.append(j)

        # close the pool
        print('Closing down the kernels...\n')
        pool.close()

        # impose maximum_distance misalignment constraints on matches
        filt_matches = []
        for match, key1, key2 in zip(matches, keypts[:],keypts[1:]):
            filteredMask = euclidMatch(match, key1, key2, maxDis)
            filt_matches.append(match[filteredMask])


        return matches, filt_matches

    #TODO: Need Better Error Handling.
    def findTransformation(self, transform, matches, processes, **kwargs):
        """ This is a Method that finds the optimal transformation between two images
        given matching features using a random sample consensus.
            Input:
                transform: skimage.transform object
                matches (list): matches found through match_features method.
                processors: Number of processors to use.
                **kwargs are passed to skimage.transform.ransac

            Output:
                Transformations.
        """

        keypts = self.features[0]

        def optimization(Pts):
            robustTrans, inliers = ransac((Pts[0], Pts[1]), transform, **kwargs)
            output = [robustTrans, inliers]
            return output

        results = [optimization(key1[match[:, 0]], key2[match[:, 1]])
                    for match, key1, key2 in zip(matches,keypts[:],keypts[1:])]

        # get Transforms and inlier matches
        transforms, trueMatches =[], []
        print('Extracting Inlier Matches with RANSAC...')
        try:
            for res in results:
                transforms.append(res[0])
                trueMatches.append(res[1])
        except np.linalg.LinAlgError:
            print('Error: Inverse of the transformation failed!!!')

        return transforms, trueMatches


    def applyTransformation(self, transforms, **kwargs):
        """ This is the method that takes the list of transformation found by findTransformation
         and applies them to the data set.

         Input:
             transforms: (list of skimage.GeoemetricTransform objects).
                     The objects must be inititated with the desired parameters.
             transformation: string, optional.
                     The type of geometric transformation to use (i.e. translation, rigid, etc..)
                     Currently, only translation is implemented.
                     default, translation.
             origin: int, optional
                     The position in the data to take as origin, i.e. don't transform.
                     default, center image in the stack.

        Output:
            Transformed images, transformations

        """
        dic = ['processors','origin','transformation']
        for key in kwargs.keys():
            if key not in dic:
                print('%s is not a parameter of this function' %(str(key)))

        processes = kwargs.get('processors', 1)
        origin = kwargs.get('origin', int(self.data.shape[0]/2))
        transformation = kwargs.get('transformation','translation')

        dset = self.data
        # For now restricting this to just translation... Straightforward to generalize to other transform objects.
        if transformation == 'translation':

            YTrans = np.array([trans.translation[0] for trans in transforms])
            XTrans = np.array([trans.translation[1] for trans in transforms])
            chainL = []
            for y, x in zip(range(0,YTrans.size+1), range(0,XTrans.size+1)):
                if y < origin:
                    ychain = -np.sum(YTrans[y:origin])
                    xchain = -np.sum(XTrans[x:origin])

                elif y > origin:
                    ychain = np.sum(YTrans[origin:y])
                    xchain = np.sum(XTrans[origin:x])
                else:
                    ychain = 0
                    xchain = 0

                chainL.append([xchain, ychain])

            chainTransforms = []
            for params in  chainL:
                T = TranslationTransform(translation = params)
                chainTransforms.append(T)

        # Just need a single function that does boths
        if transformation == 'rotation':

            rotTrans = np.array([trans.rotation for trans in transforms])
            YTrans = np.array([trans.translation[0] for trans in transforms])
            XTrans = np.array([trans.translation[1] for trans in transforms])
            chainL = []
            for x in range(0,rotTrans.size+1):
                if x < origin:
                    rotchain = -np.sum(rotTrans[x:origin])
                    ychain = -np.sum(YTrans[x:origin])
                    xchain = -np.sum(XTrans[x:origin])

                elif x > origin:
                    rotchain = np.sum(rotTrans[origin:x])
                    ychain = np.sum(YTrans[origin:x])
                    xchain = np.sum(XTrans[origin:x])
                else:
                    rotchain = 0
                    ychain = 0
                    xchain = 0

                chainL.append([rotchain, xchain, ychain])

            chainTransforms = []
            for params in  chainL:
                T = SimilarityTransform(scale = 1.0, rotation = np.deg2rad(params[0]), translation = (params[1],params[2]))
#                T = SimilarityTransform(rotation = params, translation = (0,0))
                chainTransforms.append(T)

        # Use the chain transformations to transform the dataset
        output_shape = dset[0].shape
        def warping(datum):
            imp, transform  = datum[0], datum[1]
            transimp = warp(imp, inverse_map= transform, output_shape = output_shape,
                            cval = 0, preserve_range = True)
            return transimp

        # get transformed images and pack into 3d np.ndarray
        print('Transforming Images...')
        transImages = np.copy(dset[:])

        for imp, transform, itm in zip( transImages, chainTransforms, range(0,transImages.shape[0])):
            transimp = warping([imp, transform])
            transImages[itm] = transimp
            print('Image #%i'%(itm))


        return transImages, chainTransforms

    def correlationTransformation(self, **kwargs):
        """ Uses Cross-correlation to find a translation between 2 images.
            Input:
                Processors: int, optional
                    Number of processors to use, default = 1.

            Output:
                Transformations.
        """

        processes = kwargs.get('processors', 1)

        pool = mp.Pool(processes)
        print('launching %i kernels...'%(processes))

        def register(images):
            imp1, imp2 = images[0], images[1]
            shifts, _, _ = register_translation(imp1,imp2)
            return shifts

        results = [register((imp1, imp2))
                    for imp1, imp2 in zip(self.data[:], self.data[1:])]

        return results


