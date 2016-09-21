# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 11:13:22 2016

@author: Rama K. Vasudevan, Stephen Jesse

Various helper functions for aiding loop fitting and projection
"""

from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import least_squares
from scipy.optimize import leastsq
from scipy.spatial import ConvexHull
from scipy.special import erf, erfinv


###############################################################################

def calcCentroid(Vdc,loop_vals):
    """
    Calculates the centroid of a single given loop. Uses polyogonal centroid, 
    see wiki article for details.
    
    Parameters
    -----------
    Vdc : 1D list or numpy array
        DC voltage steps
    loop_vals : 1D list or numpy array
        unfolded loop
    
    Returns
    -----------
    cent : tuple
        (x,y) coordinates of the centroid
    area : float
        geometric area
    """
    
    x_vals = np.zeros(len(Vdc)-1)
    y_vals = np.zeros(len(Vdc)-1)
    area_vals = np.zeros(len(Vdc)-1)
    
    for index in range(0,len(Vdc)-1):
        x_i = Vdc[index]
        x_i1 = Vdc[index+1]
        y_i = loop_vals[index]
        y_i1 = loop_vals[index+1]
        
        x_vals[index] = (x_i + x_i1)*(x_i*y_i1 - x_i1*y_i)
        y_vals[index] = (y_i + y_i1)*(x_i*y_i1 - x_i1*y_i)
        area_vals[index] = (x_i*y_i1 - x_i1*y_i)
    
    area = 0.50 * np.sum(area_vals)
    Cx = (1.0/(6.0*area)) * np.sum(x_vals)
    Cy = (1.0/(6.0*area)) * np.sum(y_vals)
    
    return ((Cx,Cy), area)


###############################################################################

def rotateMat(theta):
    """
    Returns the rotation matrix in 2D, given the angle theta

    Parameters
    ----------
    theta : float
        Angle in radians

    Returns
    ---------
    mat : 2D numpy array
        rotation matrix
    """
    return np.array([[np.cos(theta), -1 * np.sin(theta)], [np.sin(theta), np.cos(theta)]])


###############################################################################

def projectLoop(Vdc, amp_vec, phase_vec):
    '''
    This function projects a single loop cycle using the amplitude and phase vectors

    Parameters
    ------------
    Vdc : 1D list or numpy array
        DC voltages. vector of length N
    amp_mat : 1D numpy array
        amplitude of response
    phase_mat : 1D numpy array
        phase of response

    Returns
    ----------
    results : dictionary
        Results from projecting the provided vectors with following components

        'Projected Loop' : 1D numpy array
            projected loop matrix
        'Rotation Matrix' : tuple
            rotation angle [rad] for the projecting, as well as the offset value
        'Centroid' : M-length list of tuples
            (x,y) positions of centroids for each projected loop
        'Geometric Area' : M element long numpy array
            geometric area of the loop
    '''

    def f_min(X, p):
        plane_xyz = p[0:3]
        distance = (plane_xyz * X.T).sum(axis=1) + p[3]
        return distance / np.linalg.norm(plane_xyz)

    def residuals(params, signal, X):
        return f_min(X, params)

    Acosphi = amp_vec * np.cos(phase_vec)
    Asinphi = amp_vec * np.sin(phase_vec)

    # Fit to a plane
    # Plane equation Ax + By + Cz = D
    XYZ = np.vstack((Vdc, Acosphi, Asinphi))  # Data points in 3D

    # Inital guess of the plane
    p0 = [1, .5, -1, .5]

    # do the fitting
    sol = leastsq(residuals, p0, args=(None, XYZ))[0]
    A = sol[0]
    B = sol[1]
    C = sol[2]
    D = sol[3]

    # Choose a voltage range and y range and plot the plane with these values.i.e.,
    # plot z = (D-Ax-By)/C
    x = np.linspace(np.min(Vdc), np.max(Vdc), 200)
    y = np.linspace(np.min(Acosphi), np.max(Acosphi), 200)

    xx, yy = np.meshgrid(x, y)  # x and y limits
    zz = (-D - A * xx - B * yy) / C

    '''Number of points in the linear fit.
    This will make the offset more accurate, but 100 is reasonable.'''
    num_pt_fit = 100

    '''Plot the Acosphi/Asinphi part of the plane as well.
    This is the projection onto the y-z plane.
    Given the equation Ax + By + Cz =D, and given x is the voltage,
    we want only the By and Cz = D part.
    so, yproj = D-Cz/B and and zproj = D-By/C'''

    # Now we need to calculate the perpendicular distance to the origin
    y_proj = (-D - C * zz[:, 0]) / B
    z_proj = (-D - B * yy[:, 0]) / C

    p = np.polyfit(y_proj, z_proj, 1)  # Fit a straight line
    xdat_fit = np.linspace(y_proj.min(), y_proj.max(), num_pt_fit)
    ydat_fit = np.polyval(p, xdat_fit)

    # Find the point on the line closest to the origin

    pdist_vals = np.zeros(shape=len(xdat_fit))
    for point in range(len(xdat_fit)):
        pdist_vals[point] = np.linalg.norm(np.array([0, 0]) -
                                  np.array([xdat_fit[point], ydat_fit[point]]))

    min_point_ind = np.where(pdist_vals == pdist_vals.min())[0][0]
    min_point = np.array([xdat_fit[min_point_ind], ydat_fit[min_point_ind]])
    offset_dist = np.linalg.norm(np.array([0, 0]) - min_point)
    rot_angle = np.tan(p[0])

    # Now adjust the loop by first subtracting offset
    xdata_minus_off = Acosphi - xdat_fit[min_point_ind]
    ydata_minus_off = Asinphi - ydat_fit[min_point_ind]

    # Do the rotation
    R = rotateMat(rot_angle)
    R_orth = rotateMat(np.pi + rot_angle)

    ydat_new = np.dot(R, [xdata_minus_off, ydata_minus_off])  # This has the rotated components
    ydat_new_orth = np.dot(R_orth, [xdata_minus_off,
                                    ydata_minus_off])  # This has rotated components, but using the other angle.

    # taking only the first index, since this is all that was being used.
    ydat_new = ydat_new[0, :]
    ydat_new_orth = ydat_new_orth[0, :]

    """
    Now determine whether the correct one is clockwise or counter clockwise
    The data that we need is ydat_new and ydat_new_orth
    We are going to calculate the center of mass of the rotated loop
    Using definition of centroid of a polygon (see wikipedia https://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon)
    """

    # Centroid calculation also gives geometric area. If the area is positive then the loop rotates the correct way.
    # If the area is negative it rotates the 'wrong' way.

    # print ydat_new.shape, ydat_new_orth.shape
    c_point, geo_area_loop = calcCentroid(Vdc, ydat_new)
    c_point_orth, geo_area_loop_orth = calcCentroid(Vdc, ydat_new_orth)

    pr_vec = np.zeros(shape=(len(ydat_new)))

    # Based on the above we can see that the blue curve gives the correct rotation angle.
    if geo_area_loop > 0:
        pr_vec = ydat_new  # [0,:]
        centroid = c_point
        geo_area = geo_area_loop
        rot_angle = rot_angle
    else:
        pr_vec = ydat_new_orth  # [0,:]
        centroid = c_point_orth
        geo_area = geo_area_loop_orth
        rot_angle = rot_angle + np.pi

    results = {'Projected Loop': pr_vec.ravel(), 'Rotation Matrix': (rot_angle, offset_dist),
               'Centroid': centroid, 'Geometric Area': geo_area}  # Dictionary of Results from projecting

    return results
    
###############################################################################

def loopFitFunction(V,coef_vec):
    """
    9 parameter fit function
    
    Parameters
    -----------
    V : 1D numpy array or list
        DC voltages
    coef_vec : 1D numpy array or list
        9 parameter coefficient vector
        
    Returns
    ---------
    F : 2D numpy array 
        function values
    """
    
    a = coef_vec[:5]
    b = coef_vec[5:]
    d = 1000
    
    V1 = V[:(len(V)/2)]
    V2 = V[(len(V)/2):]
    
    g1 = (b[1]-b[0])/2*(erf((V1-a[2])*d)+1)+b[0]
    g2 = (b[3]-b[2])/2*(erf((V2-a[3])*d)+1)+b[2]

    Y1 = ( g1 * erf((V1-a[2])/g1) + b[0] )/(b[0]+b[1])
    Y2 = ( g2 * erf((V2-a[3])/g2) + b[2] )/(b[2]+b[3])

    F1 = a[0] + a[1]*Y1 + a[4]*V1
    F2 = a[0] + a[1]*Y2 + a[4]*V2
    
    F = np.hstack((F1,F2))
    return F    

###############################################################################

def loopFitJacobian(V, coef_vec):
    """
    Jacobian of 9 parameter fit function

    Parameters
    -----------
    V : 1D numpy array or list
        DC voltages
    coef_vec : 1D numpy array or list
        9 parameter coefficient vector

    Returns
    ---------
    F : 2D numpy array
        function values
    """

    a = coef_vec[:5]
    b = coef_vec[5:]
    d = 1000

    J = np.zeros([len(V),9], dtype=np.float32)

    V1 = V[:(len(V) / 2)]
    V2 = V[(len(V) / 2):]

    g1 = (b[1] - b[0]) / 2 * (erf((V1 - a[2]) * d) + 1) + b[0]
    g2 = (b[3] - b[2]) / 2 * (erf((V2 - a[3]) * d) + 1) + b[2]

# Some useful fractions
    oosqpi = 1.0/np.sqrt(np.pi)
    oob01 = 1.0 / (b[0] + b[1])
    oob23 = 1.0 / (b[2] + b[3])

# Derivatives of g1 and g2
    dg1a2 = -(b[1] - b[0]) / oosqpi * d * np.exp(-(V1 - a[2]) ** 2)
    dg2a3 = -(b[3] - b[2]) / oosqpi * d * np.exp(-(V2 - a[3]) ** 2)
    dg1b0 = -0.5 * (erf(V1 - a[2]) * d + 1)
    dg1b1 = -dg1b0
    dg2b2 = -0.5 * (erf(V2 - a[3]) * d + 1)
    dg2b3 = -dg2b2

    Y1 = oob01 * (g1 * erf((V1-a[2])/g1) + b[0])
    Y2 = oob23 * (g2 * erf((V2-a[3])/g2) + b[2])

# Derivatives of Y1 and Y2
    dY1g1 = oob01 * (erf((V1 - a[2]) / g1) - 2 * oosqpi * np.exp(-(V1 - a[2])**2 / g1**2))
    dY2g2 = oob23 * (erf((V2 - a[4]) / g2) - 2 * oosqpi * np.exp(-(V2 - a[4])**2 / g2**2))

    dY1a2 = -2 * oosqpi * oob01 * np.exp(-(V1 - a[2])**2 / g1**2)

    dY2a3 = -2 * oosqpi * oob01 * np.exp(-(V2 - a[3])**2 / g2**2)

    dY1b0 = oob01 * (1 - Y1)
    dY1b1 = -oob01 * Y1

    dY2b2 = oob23 * (1 - Y2)
    dY2b3 = -oob23 * Y2

    '''
    Jacobian terms
    '''
    zeroV = np.zeroes_like(V1)
# Derivative with respect to a[0] is always 1
    J[:, 0] = 1

# Derivative with respect to a[1] is different for F1 and F2
    J[:, 1] = np.hstack(Y1, Y2)

# Derivative with respect to a[2] is zero for F2, but not F1
    J21 = a[1] * (dY1g1 * dg1a2 + dY1a2)
    J22 = zeroV
    J[:, 2] = np.hstack(J21, J22)

# Derivative with respect to a[3] is zero for F1, but not F2
    J31 = zeroV
    J32 = a[1] * (dY2g2 * dg2a3 + dY2a3)
    J[:, 3] = np.hstack(J31, J32)

# Derivative with respect to a[4] is V
    J[:, 4] = V

# Derivative with respect to b[0] is zero for F2, but not F1
    J51 = a[1] * (dY1g1 * dg1b0 + dY1b0)
    J52 = zeroV
    J[:, 5] = np.hstack(J51, J52)

# Derivative with respect to b[1] is zero for F2, but not F1
    J61 = a[1] * (dY1g1 * dg1b1 + dY1b1)
    J62 = zeroV
    J[:, 6] = np.hstack(J61, J62)

# Derivative with respect to b[2] is zero for F1, but not F2
    J71 = zeroV
    J72 = a[1] * (dY2g2 * dg2b2 + dY2b2)
    J[:, 7] = np.hstack(J71, J72)

# Derivative with respect to b[3] is zero for F1, but not F2
    J81 = zeroV
    J82 = a[1] * (dY2g2 * dg2b3 + dY2b3)
    J[:, 8] = np.hstack(J81, J82)

    return J

    
###############################################################################


def getSwitchingCoefs(loop_centroid,loop_coef_vec):
    """
    Parameters
    -----------
    loop_centroid : tuple or 1D list or numpy array
        [x,y] position of loop centroid
    loop_coef_vec : 1D list or numpy array
        a 9 length vector of coefficients describing the loop via the 9 loop parmeter fit
    
    Returns
    --------
    switching_labels_dict : dictionary
        switching labels and values
    """
    #calculate nucleation bias  
    anv = loop_coef_vec[0:5]
    bnv = loop_coef_vec[5:]
    nuc_threshold = .97
    
    lc_x = loop_centroid[0]
    
    #Upper Branch    
    B = bnv[2]
    nuc_v01a = B*erfinv( (nuc_threshold*bnv[2] + nuc_threshold*bnv[3] - bnv[2]) /B ) - abs(anv[2]) 
    
    B = bnv[3]
    nuc_v01b = B*erfinv( (nuc_threshold*bnv[2] + nuc_threshold*bnv[3] - bnv[2]) /B ) -abs(anv[2])
  
    
    #Choose the voltage which is closer to the centroid x value.
    if abs(nuc_v01a-lc_x)<=abs(nuc_v01b-lc_x):
        nuc_1 = nuc_v01a
    else:
        nuc_1 = nuc_v01b
        
    if np.isinf(nuc_1)==True: nuc_1 = 0
    
    ##Lower Branch
    B = bnv[0]
    nuc_v02a = B*erfinv( ((1-nuc_threshold)*bnv[0] + (1-nuc_threshold)*bnv[1] - bnv[0])/B ) + abs(anv[3]) 
    
    B = bnv[1]
    nuc_v02b = B*erfinv( ((1-nuc_threshold)*bnv[0] + (1-nuc_threshold)*bnv[1] - bnv[0])/B ) + abs(anv[3])
    
    if abs(nuc_v02a-lc_x)<=abs(nuc_v02b-lc_x):
        nuc_2 = nuc_v02a
    else:
        nuc_2 = nuc_v02b
        
    if np.isinf(nuc_2)==True: nuc_2 = 0

    #calculate switching parameters
    switching_coef_vec = np.zeros(shape=(9,1))
    
    switching_coef_vec[0] = loop_coef_vec[2]
    switching_coef_vec[1] = loop_coef_vec[3]
    switching_coef_vec[2] = (loop_coef_vec[3]+loop_coef_vec[2])/2
    switching_coef_vec[3] = loop_coef_vec[0]+loop_coef_vec[1]
    switching_coef_vec[4] = loop_coef_vec[0]
    switching_coef_vec[5] = loop_coef_vec[1]
    switching_coef_vec[6] = abs((loop_coef_vec[2]-loop_coef_vec[3])*abs(loop_coef_vec[1]))
    switching_coef_vec[7] = nuc_1
    switching_coef_vec[8] = nuc_2

    switching_labels_dict = {'V-': switching_coef_vec[0],'V+':switching_coef_vec[1],
                             'Imprint':switching_coef_vec[2], 'R+':switching_coef_vec[3],
                            'R-':switching_coef_vec[4], 'Switchable Polarization':switching_coef_vec[5],
                            'Work of Switching':switching_coef_vec[6],
                            'Nucleation Bias 1':nuc_1,  'Nucleation Bias 2':nuc_2}
    
    return switching_labels_dict

##############################################################################

def generateGuess(Vdc, pr_vec, show_plots=False):
    '''
    Given a single unfolded loop and centroid return the intial guess for the fitting.
    We generate most of the guesses by looking at the loop centroid and looking
    at the nearest intersection points with the loop, which is a polygon.

    Parameters
    -----------
    Vdc : 1D numpy array
        DC offsets
    pr_vec : 1D numpy array
        Piezoresponse or unfolded loop
    show_plots : Boolean (Optional. Default = False)
        Whether or not the plot the convex hull, centroid, intersection points

    Returns
    -----------------
    init_guess_coef_vec : 1D Numpy array
        Fit guess coefficient vector
    '''

    points = np.transpose(np.array([np.squeeze(Vdc), pr_vec]))  # [points,axis]

    geom_centroid, geom_area = calcCentroid(points[:, 0], points[:, 1])

    hull = ConvexHull(points)

    ''' Now we need to find the intersection points on the N,S,E,W
    the simplex of the complex hull is essentially a set of line equations.
    We need to find the two lines (top and bottom) or (left and right) that
    interect with the vertical / horizontal lines passing through the geometric centroid'''

    def findIntersection(A, B, C, D):
        '''
        Finds the coordinates where two line segments intersect

        Parameters
        ------------
        A, B, C, D : Tuple or 1D list or 1D numpy array
            (x,y) coordinates of the points that define the two line segments AB and CD

        Returns
        ----------
        obj : None or tuple
            None if not intersecting. (x,y) coordinates of intersection
        '''

        def ccw(A, B, C):
            '''Credit - StackOverflow'''
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def line(p1, p2):
            '''Credit - StackOverflow'''
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0] * p2[1] - p2[0] * p1[1])
            return A, B, -C

        def intersection(L1, L2):
            '''
            Finds the intersection of two lines (NOT line segments).
            Credit - StackOverflow
            '''
            D = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]
            if D != 0:
                x = Dx / D
                y = Dy / D
                return x, y
            else:
                return None

        if (ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)) == False:
            return None
        else:
            return intersection(line(A, B), line(C, D))

    # start and end coordinates of each line segment defining the convex hull
    outline_1 = np.zeros((hull.simplices.shape[0], 2), dtype=np.float)
    outline_2 = np.zeros((hull.simplices.shape[0], 2), dtype=np.float)
    for index, pair in enumerate(hull.simplices):
        outline_1[index, :] = points[pair[0]]
        outline_2[index, :] = points[pair[1]]

    '''Find the coordinates of the points where the vertical line through the
    centroid intersects with the convex hull'''
    Y_intersections = []
    for pair in xrange(outline_1.shape[0]):
        x_pt = findIntersection(outline_1[pair], outline_2[pair],
                                [geom_centroid[0], hull.min_bound[1]],
                                [geom_centroid[0], hull.max_bound[1]])
        if type(x_pt) != type(None):
            Y_intersections.append(x_pt)

    '''Find the coordinates of the points where the horizontal line through the
    centroid intersects with the convex hull'''
    X_intersections = []
    for pair in xrange(outline_1.shape[0]):
        x_pt = findIntersection(outline_1[pair], outline_2[pair],
                                [hull.min_bound[0], geom_centroid[1]],
                                [hull.max_bound[0], geom_centroid[1]])
        if type(x_pt) != type(None):
            X_intersections.append(x_pt)

    min_Y_intercept = min(Y_intersections[0][1], Y_intersections[1][1])
    max_Y_intercept = max(Y_intersections[0][1], Y_intersections[1][1])

    # Only the first four parameters use the information from the intercepts
    init_guess_coef_vec = np.zeros(shape=(9))
    init_guess_coef_vec[0] = min_Y_intercept
    init_guess_coef_vec[1] = max_Y_intercept - min_Y_intercept
    init_guess_coef_vec[2] = max(X_intersections[0][0], X_intersections[1][0])
    init_guess_coef_vec[3] = min(X_intersections[0][0], X_intersections[1][0])
    init_guess_coef_vec[4] = 0
    init_guess_coef_vec[5] = .5
    init_guess_coef_vec[6] = .2
    init_guess_coef_vec[7] = 1
    init_guess_coef_vec[8] = .2

    if show_plots:
        fig, ax = plt.subplots()
        ax.plot(points[:, 0], points[:, 1], 'o')
        ax.plot(geom_centroid[0], geom_centroid[1], 'r*')
        ax.plot([geom_centroid[0], geom_centroid[0]], [hull.max_bound[1], hull.min_bound[1]], 'g')
        ax.plot([hull.min_bound[0], hull.max_bound[0]], [geom_centroid[1], geom_centroid[1]], 'g')
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], 'k')
        ax.plot(X_intersections[0][0], X_intersections[0][1], 'r*')
        ax.plot(X_intersections[1][0], X_intersections[1][1], 'r*')
        ax.plot(Y_intersections[0][0], Y_intersections[0][1], 'r*')
        ax.plot(Y_intersections[1][0], Y_intersections[1][1], 'r*')
        ax.plot(Vdc, loopFitFunction(Vdc, init_guess_coef_vec))

    return init_guess_coef_vec


###############################################################################

def fitLoop(Vdc_shifted, pr_shifted, guess):
    '''
    Given a single unfolded loop returns the results of the least squares fitting

    Parameters
    ----------
    Vdc_shifted : 1D numpy array
        DC voltage values shifted by one quarter, as this is requirement for the fit
    pr_shifted : 1D numpy array
        unfolded loop shifted by one quarter, as this is requirement for the fit
    guess : 1D numpy array
        9 parameters for the fit guess

    Returns
    --------
    plsq : least squares fit structure
        Contains fit results, fit parameters, covariance, etc.
    criterion_values : tuple
        (AIC (loop fit), BIC(loop fit), AIC(line fit), BIC(line fit))
        AIC and BIC values for loop and line fits
    pr_fit_vec : 1D list or numpy array
        fit result values, ie. evaluation of f(V).
    '''

    def loopResiduals(p, y, x):
        err = y - loopFitFunction(x, p)
        return err

    def loopJacResiduals(p, y, x):
        Jerr = -loopFitJacobian(x, p)
        return Jerr

    # We may need to think about these bounds further, but these will work for now.

    lb = ([-1E3, -1E3, -1E3, -1E3, -10, -100, -100, -100, -100])  # Lower Bounds
    ub = ([1E3, 1E3, 1E3, 1E3, 10, 100, 100, 100, 100])  # Upper Bounds

    xdata = Vdc_shifted.ravel()
    ydata = pr_shifted.ravel()

    '''Do the fitting. Least Squares fit. Using more accurate determination of
    Jacobian. This is slower, but will be necessary initially for generating the
    guesses (see below)'''
    # plsq = least_squares(loopResiduals, guess.ravel(), args=(ydata, xdata), jac='3-point',
    #                      max_nfev=1E4,bounds=((lb,ub)),loss='soft_l1')
    plsq = least_squares(loopResiduals, guess.ravel(), args=(ydata, xdata), jac=loopJacResiduals,
                         max_nfev=1E4, bounds=((lb, ub)), loss='soft_l1')
    pr_fit_vec = loopFitFunction(xdata, plsq.x)

    '''Here we compare the values of the information criterion, for the whole loop fit and a simple linear fit
    We use both the AIC and BIC creterion metrics to compare which is better
    Lower values (even negative) are better than higher values).'''

    N = len(xdata)
    sd_loop = np.std((ydata - pr_fit_vec))
    df_loop = 8  # degrees of freedom
    df_line = 1

    LL = np.sum(stats.norm.logpdf(ydata, loc=pr_fit_vec, scale=sd_loop))  # log liklihood estimation
    AIC_loop = 2.0 * df_loop - 2.0 * LL  # calculate AIC
    BIC_loop = -2.0 * LL + df_loop * np.log(N)  # calculate BIC

    lin_fit = np.polyfit(xdata, ydata, 1)

    sd_line = np.std((ydata - np.polyval(lin_fit, xdata)))
    L2 = np.sum(stats.norm.logpdf(ydata, loc=np.polyval(lin_fit, xdata), scale=sd_line))

    AIC_line = 2.0 * df_line - 2.0 * L2
    BIC_line = -2.0 * L2 + df_line * np.log(N)

    criterion_values = (AIC_loop, BIC_loop, AIC_line, BIC_line)

    return plsq, criterion_values, pr_fit_vec