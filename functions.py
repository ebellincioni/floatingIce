import numpy as np
from matplotlib import pyplot as plt
import glob
# from scipy.stats import linregress
from shapely.geometry import Polygon, Point
from scipy.special import erf
from scipy.optimize import curve_fit 
# import matplotlib.image as mpimg
import cv2
from scipy.integrate import solve_ivp,odeint
# from matplotlib.colors import hsv_to_rgb
from simplefit import fit # https://gist.github.com/aminnj/37ab33533b07b7007623ae278c5d5797
import uncertainties
from uncertainties import unumpy
import circle_fit
import geopandas as gpd
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from uncertainties import ufloat,unumpy
from lmfit import minimize, Parameters, Parameter, report_fit

import matplotlib as mpl
font = {'family' : 'Helvetica Neue',
        'weight' : 'normal',
        'size'   : 12}

mpl.rc('font', **font)

g = 9.81 # m/s2, gravity
beta = 1e-4 # degC-1, thermal coefficient expansion water
deltaT = 20.5 # degC, melting-ambient
alpha = 0.133e-6 # m2/s, thermal diffiusivity water
#alpha = 1.02e-6 # m2/s, thermal diffiusivity ice
nu = 1.4e-6 # m2/s, kinematic viscosity water
Pr = 10 # variable
kappa = 580e-3 # W/Km, thermal conductivity water
T_in = 16 # degC, initial negative temperature ice
rho_ice = 917 # kg/m3, density ice
L_f = 334e3 # J/kg, latent heat fusion water
c_s = 4186 # J/kgK, specific heat water 

def gradient_with_errors(uarray, dx):
    """
    Calculates the gradient of a 1D array of ufloat values (with uncertainties).
    
    Parameters:
        uarray (numpy array of ufloat): The array of uncertain values.
        dx (float): The spacing between points in the array (assumed constant).
        
    Returns:
        numpy array of ufloat: The gradient with propagated uncertainties.
    """
    # Initialize the gradient array
    grad_uarray = []

    # Loop over elements to calculate the gradient with propagated uncertainties
    for i in range(len(uarray)):
        if i == 0:
            # Forward difference for the first element
            diff = uarray[i + 1] - uarray[i]
            grad = diff / dx
        elif i == len(uarray) - 1:
            # Backward difference for the last element
            diff = uarray[i] - uarray[i - 1]
            grad = diff / dx
        else:
            # Central difference for the middle elements
            diff = uarray[i + 1] - uarray[i - 1]
            grad = diff / (2 * dx)
        
        grad_uarray.append(grad)

    return np.array(grad_uarray)

def sliding_window_polyfit(x, y, window_size=10, poly_order=2):
    """
    Fits a second-order polynomial in a sliding window over the data and calculates
    the error for both the fit and the gradient.
    
    Parameters:
    x (array-like): The x-values of the data points.
    y (array-like): The y-values of the data points.
    window_size (int): The size of the sliding window.
    poly_order (int): The order of the polynomial (default is 2 for quadratic).
    
    Returns:
    list of ndarray: List of polynomial coefficients for each window.
    ndarray: Array of fitted y-values only at the central indices.
    ndarray: Array of fitting errors (MSE) only at the central indices.
    ndarray: Array of gradients at the central indices.
    ndarray: Array of gradient errors at the central indices.
    """
    poly_coeffs = []
    y_fitted = np.full_like(y, np.nan, dtype=np.double)  # Array to hold central fitted values
    fit_errors = np.full_like(y, np.nan, dtype=np.double)  # Array to hold errors at central points
    gradient = np.full_like(y, np.nan, dtype=np.double)  # Array to hold gradient at central points
    gradient_errors = np.full_like(y, np.nan, dtype=np.double)  # Array to hold gradient errors
    
    # Loop over data with sliding windows
    for i in range(len(x) - window_size + 1):
        # Select data within the window
        x_window = x[i:i + window_size]
        y_window = y[i:i + window_size]
        
        # Fit a polynomial of the given order to the window data
        coeffs = np.polyfit(x_window, y_window, poly_order)
        poly_coeffs.append(coeffs)
        
        # Generate fitted values for the current window
        y_fit = np.polyval(coeffs, x_window)
        
        # Calculate the mean squared error (MSE) for the current window
        error = np.mean((y_window - y_fit) ** 2)
        
        # Calculate the gradient (slope) of the fit (1st derivative of the polynomial)
        gradient_value = 2 * coeffs[0] * x_window[window_size // 2] + coeffs[1]
        
        # Estimate the gradient error as the standard deviation of residuals divided by the range of x values
        gradient_error = np.sqrt(np.mean((y_window - y_fit) ** 2)) / (x_window[-1] - x_window[0])
        
        # Assign the results to the central index
        central_index = i + window_size // 2
        y_fitted[central_index] = y_fit[window_size // 2]
        fit_errors[central_index] = error
        gradient[central_index] = gradient_value
        gradient_errors[central_index] = gradient_error
    
    return poly_coeffs, y_fitted, fit_errors, gradient, gradient_errors

def remove_and_interpolate_outliers(data, max_relative_decrease=0.1):
    """
    Removes isolated outliers from a 1D monotonically decreasing array by replacing
    them with linearly interpolated values if the decrease is too sharp.
    
    Parameters:
    data (array-like): The input 1D array, expected to be monotonically decreasing.
    max_relative_decrease (float): The maximum allowed relative decrease between consecutive elements.
    
    Returns:
    ndarray: Array with isolated outliers replaced by interpolated values.
    """
    # Convert to a NumPy array for easier manipulation
    data = np.array(data, dtype=np.float64)
    cleaned_data = data.copy()  # Copy the original array to preserve it
    
    # Loop through the array to identify and interpolate outliers
    for i in range(1, len(data) - 1):
        previous_value = cleaned_data[i - 1]
        current_value = data[i]
        next_value = data[i + 1]
        
        # Calculate the relative decrease with the previous value
        relative_decrease = (previous_value - current_value) / previous_value
        
        # Check if current value is an outlier based on the relative decrease threshold
        if relative_decrease > max_relative_decrease:
            # Interpolate as the average of previous and next valid values
            interpolated_value = (previous_value + next_value) / 2
            cleaned_data[i] = interpolated_value
            # print(f"Outlier detected at index {i}: {current_value}, replaced with {interpolated_value}")
    
    return cleaned_data




def uNu_fit_per(a_params,a_sigma,p_params,p_sigma,t,a0,p0,T_water):
    """Function that returns the Nusselt number coming from a fit through areas and perimeters of an experiment. 

    Args:
        a_params (lmfit Parameters): best fit parameters for the areas
        p_params (lmfit Parameters): best fit parameters for the perimeters
        a_sigma (array) : array of stds for areas
        p_sigma (array) : array of stds for perimeters
        t (np.array): time
        a0 (float): initial area
        p0 (float): initial perimeter
        T_water (float): water temperature

    Returns:
        uarray: Nusselt numbers with error. 
    """    
    a_a = a_params['a']['value']
    a_b = a_params['b']['value']
    a_c = a_params['c']['value']
    a = unumpy.uarray([i for i in a_a*t**3 + a_b*t**2 + a_c*t + a0], a_sigma)
    p_a = p_params['a']['value']
    p_b = p_params['b']['value']
    p_c = p_params['c']['value']
    p = unumpy.uarray([i for i in p_a*t**3 + p_b*t**2 + p_c*t + p0], p_sigma) # perimeter
    try:
        # the reason why I'm doing this is because there are some values of L-\sigma_L that cross zero. And the sqrt screws up. This obviously is not true for L.nominal_value
        Adot = (3*a_a*t**2+2*a_b*t+a_c) * unumpy.sqrt((a_a*t**3 + a_b*t**2 + a_c*t + a0)/np.pi) 
    except ValueError:
        Adot = (3*a_a.nominal_value*t**2+2*a_b.nominal_value*t+a_c.nominal_value) * np.sqrt((a_a.nominal_value*t**3 + a_b.nominal_value*t**2 + a_c.nominal_value*t + a0)/np.pi)
    # print(Adot)
    Adot = gradient_with_errors(a,t[1]-t[0])*unumpy.sqrt(a/np.pi)
    # print(Adot)
    h = -(Adot*2)*rho_ice*L_f/(p*T_water)
    return h/kappa

def pickFromDistr(mean,std):
    pick = np.random.normal(mean,std)
    return pick
def bootstrap(result):
    bsParams = np.zeros(len(result['params']))
    for ip,param in enumerate(result['params']):
        # print(result['params'][param])
        bsParams[ip] = pickFromDistr(result['params'][param]['value'],result['params'][param]['error'])
    return bsParams

def uNu_areas(A,T_water, dt):
    """Returns an approximated value of the Nusselt number that accounts for the latent and sensitive heat.
    The parameter passed is the area of the cylinder. 

    Args:
        A (float): area of cylinder [m2]
        T_water (float): bulk water temperature [degC]
        dt (float): time interval [s]

    Returns:
        float: value of Nusselt number
    """   
    h = -(gradient_with_errors(A,dt))*rho_ice*(L_f+c_s*T_in)/(np.pi*T_water)
    return h/kappa

def polynomial_fitting(xdata,ydata,degree,boolPlot):
    """Fits a polynomial to data. 

    Args:
        xdata (np.array/list): data that will be on x axis
        ydata (np.array/list): data to be fitted (on y axis)
        degree (int): degree of polynomial
        boolPlot (bool): wheter to plot a line with the output or not
    Returns:
        np.array: parameters of best fitting function
        np.poly1d: polynomial element of numpy
    """    
    fit_par = np.polyfit(xdata, ydata,degree)
    yfit = np.poly1d(fit_par)
    if boolPlot: plt.plot(xdata, yfit(xdata),c='k',linestyle='--')
    return fit_par,yfit

def running_mean(x,N):
    """Function that returns the running mean on the array x with window N

    Args:
        x (np.array): input array
        N (int): window size

    Returns:
        np.array: averaged array (len = len(x)-N+1)
    """    
    return np.convolve(x, np.ones(N)/N, mode='valid')

def sigmoid_fitting(xdata,ydata,exp_y0,boolPlot):
    """Fits a sigmoid function (sigmoid) to data. 

    Args:
        xdata (np.array/list): data that will be on x axis
        ydata (np.array/list): data to be fitted (on y axis)
        exp_y0 (float): known (experimental) value for the ydata[0]
        boolPlot (bool): wheter to plot a line with the output or not
    Returns:
        np.array: best fitting function
    """    
    p0 = [-ydata[0], xdata[-1]/2,exp_y0*1e2/xdata[-1],ydata[0]] 
    popt, _ = curve_fit(sigmoid, xdata, ydata,p0,maxfev=int(1e5))
    if boolPlot: plt.plot(xdata, sigmoid(xdata,popt[0],popt[1],popt[2],popt[3]),c='k',linestyle='--')
    return sigmoid(xdata,popt[0],popt[1],popt[2],popt[3])

def sigmoid(x, L ,x0, k, b):
    """Returns a sigmoid function 

    Args:
        x (np.array): input array
        L (float): function parameter
        x0 (float): function parameter
        k (float): function parameter
        b (float): function parameter

    Returns:
        np.array: output array
    """    
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

def gauss(x, H, A, x0, sigma): 
    """Returns a gaussian function

    Args:
        x (np.array): input array
        H (float): function parameter
        A (float): function parameter
        x0 (float): function parameter
        sigma (float): function parameter

    Returns:
        np.array: output array
    """    
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def correct_outliers(y,precision=0.1):
    """Corrects outliers by interpolating between them. Outliers are defined where there is a change of more than a certain fraction in the normalised gradient. 
    This function can only handle one outlier.

    Args:
        y (np.array): input array
        precision (float) : between 0 and 1, max allowed change in the gradient

    Returns:
        np.array: array without outliers.
    """    
    outliers = np.where(abs(np.gradient(y/y[0]))>precision)[0]
    if len(outliers)==1:
        # only works if there is one outlier
        y[np.mean(outliers,dtype=int)] = np.nan
        y = np.array(y)
        nans,x = nan_helper(y)
        y[nans] = np.interp(x(nans),x(~nans),y[~nans])
    return y

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def rotate(points, origin, degrees):
    """Rotates a polygon (list of x,y-coordinates) about an origin of an angle in degrees

    Args:
        points (np.array/list): points coordinates
        origin (np.array/list): coordinate center of rotation
        degrees (float): degrees of rotation

    Returns:
        np.array: coordinates of rotated points
    """    
    radians = np.deg2rad(degrees)
    x,y = points[:,0],points[:,1]
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return np.vstack((qx, qy)).T

def PolyArea(x,y):
    """Returns area of a polygon given the x,y-coordinates of its points

    Args:
        x (np.array): x-coordinates
        y (np.array): y-coordinates

    Returns:
        _type_: _description_
    """    
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def Kell_rho(T):
    """Returns density of PURE water at temperature T according to the Kell function (5th order polynomial fit)

    Args:
        T (float): temperature [degC]

    Returns:
        float: density of water [kg/m3]
    """    
    a = -2.8054253e-10
    b = 1.0556302e-7
    c = -4.6170461e-5
    d = -0.0079870401
    e = 16.945176
    f = 999.83952
    g = 0.01687985
    rho = (((((a*T+b)*T+c)*T+d)*T+e)*T+f) / (1+g*T)
    return rho

def MH_rho(T,S=0):
    """Returns density of salty water at temperature T, with salinity S according to the formula in Millero, Huang,2009. 
    That formula is calculates the additional density due to the salts. The pure water density is calculated from the Kell formula. 

    Args:
        T (float): temperature [degC]
        S (float): water salinity [g/l], Defaults to 0 g/l.

    Returns:
        float: density of water [kg/m3]
    """   
    a0,a1,a2,a3,a4,a5 = 8.211458E-01,-3.959680E-03,7.305182E-05,-8.282446E-07,5.386657E-09,0
    A = a0 + a1*T + a2*T**2 + a3*T**3 + a4*T**4 + a5*T**5
    b0,b1,b2 = -6.058307E-03,8.265457E-05,-1.077747E-06
    B = b0 + b1*T + b2*T**2
    C = 5.265280E-04
    rho = A*S + B*S**1.5 + C*S**2 + Kell_rho(T)
    return rho

def uNu_fit(params,t,r0,T_water):
    # if len(params) != 3: return
    a = ufloat(params['a']['value'],params['a']['error'])
    b = ufloat(params['b']['value'],params['b']['error'])
    c = ufloat(params['c']['value'],params['c']['error'])
    rrdot = c*r0 + c**2*t + 2*b*r0*t + 3*b*c*t**2 + 3*a*r0*t**2 + 2*b**2*t**3 + 4*a*c*t**3 + 5*a*b*t**4 + 3*a**2*t**5
    Adot = 2*np.pi*rrdot # the dt is already included in the fit
    h = -(Adot)*rho_ice*(L_f+c_s*T_in)/(np.pi*T_water)
    return h/kappa

def Nu_areas(A,T_water, dt):
    """Returns an approximated value of the Nusselt number that accounts for the latent and sensitive heat.
    The parameter passed is the area of the cylinder. 

    Args:
        A (float): area of cylinder [m2]
        T_water (float): bulk water temperature [degC]
        dt (float): time interval [s]

    Returns:
        float: value of Nusselt number
    """   
    h = -(np.gradient(A)/dt)*rho_ice*(L_f+c_s*T_in)/(np.pi*T_water)
    return h/kappa

    # different version that considers perimeter (it doesn't change much...)
    # h = -(np.gradient(A)/dt)*rho_ice*(L_f+c_s*T_in)/(T_water*(np.sqrt(A)+2*A))
    # return h*np.sqrt(A)/kappa

def Ra(r,T1,T2):
    """Calculates Rayleigh number for a cylinder in water, given radius and two temperatures. The density is calculated according to the Kell's formula.

    Args:
        r (float): cylinder's radius [m]
        T1 (float): temperature 1 [degC]
        T2 (float): temperature 2 [degC]

    Returns:
        float: Rayleigh number
    """    
    Tavg = np.mean([T1,T2])
    rho = Kell_rho(Tavg)
    deltaRho = abs(Kell_rho(T1)-Kell_rho(T2))
    return g*deltaRho*(2*r)**3/(alpha*nu*rho)

def Ra_salinity(r,T1,T2,S=0):
    """Calculates Rayleigh number for a cylinder in (possibly salty) water, given radius,salinity, and two temperatures. 
    The density is calculated according to Millero and Huang, 2009.

    Args:
        r (float): cylinder's radius [m]
        S (float,optional): water salinity [g/l], Defaults to 0 g/l.
        T1 (float): temperature 1 [degC]
        T2 (float): temperature 2 [degC]

    Returns:
        float: Rayleigh number
    """    
    T_min = min(T1,T2)
    T_max = max(T1,T2)
    rho1 = MH_rho(T_min,S=0) # fresh, cold water
    rho2 = MH_rho(T_max,S=S)
    rho_avg = np.mean((rho1,rho2))

    deltaRho = abs(rho1-rho2)
    return g*deltaRho*(2*r)**3/(alpha*nu*rho_avg)

def Ra_heavyWater(r,T1):
    """Calculates Rayleigh number for a heavy water cylinder in fresh water, given radius and the temperature of the "light" water. The heavy water is assumed to be at 0degC.
    The density is calculated according to Millero and Huang, 2009.

    Args:
        r (float): cylinder's radius [m]
        T1 (float): temperature of bulk [degC]

    Returns:
        float: Rayleigh number
    """    
    rho1 = MH_rho(T1,S=0) # bulk water
    rho2 = .71*1106+.29*MH_rho(0.,0.) # heavy water melt
    rho_avg = np.mean((rho1,rho2))

    deltaRho = abs(rho1-rho2)
    return g*deltaRho*(2*r)**3/(alpha*nu*rho_avg)

def Nu(Ray):
    """Calculates Nusselt number for a cylinder in water given Rayleigh number and Prandtl number (not function variable). Formula from Churchill and Chu, 1975

    Args:
        Ray (float): Rayleigh number

    Returns:
        float: Nusselt number
    """    
    return (0.6 + (0.387*Ray**(1/6)) / ( ( 1 + (0.559/Pr)**(9/16) ) ** (16/9) ) )**2 

def h(Nu,r):   
    return kappa*Nu/(2*r)

def drdt(h,r,t):
    T_0 = -T_in
    T_star = T_0 + (deltaT-T_0)*.5*erf(r/np.sqrt(alpha*t))
    dT_stardt = (deltaT-T_0)*.5 * np.exp(-r**2/(alpha*t)) / np.sqrt(alpha*t)
    return -(h*deltaT+kappa*dT_stardt)/(rho_ice*(L_f+c_s*(-T_star)))

def evolution(r_0):
    timesteps = 1600 #s
    radii = [r_0,]
    r = r_0
    for timestep in range(1,timesteps):
        # timestep is in seconds
        Ra_ = Ra(r,-16,20)
        Nu_ = Nu(Ra_)
        h_ = h(Nu_,r)
        drdt_ = drdt(h_,r,timestep)
        # print(Ra_,Nu_,h_,drdt_)
        r = r + drdt_
        radii.append(r)
    return radii

def makePolygonBelow(polygon, level):
    """Draws the part of a polygon which is below a horizontal level. Returns a list of coordinates. 

    Args:
        polygon (np.array): polygon coordinates
        level (float): horizontal level 

    Returns:
        np.array: coordinates of polygon below level
    """    
    newY = np.where(polygon[:,1]<level,polygon[:,1],level)
    polygonBELOW = polygon.copy()
    polygonBELOW[:,1] = newY
    return polygonBELOW



def findWL(polygon, densityRatio=.917, epsilon = 0.001):
    """Finds the water level for an ice polygon immersed in water, with prescribed precision. 

    Args:
        polygon (np.array): polygon coordinates
        densityRatio (float, optional): density ratio of the two fluids. Defaults to .917, ratio of ice/fresh water.
        epsilon (float, optional): precisio of calculation. Defaults to 0.001.

    Returns:
        float: water level of the immersed polygon
    """    
    originalArea = PolyArea(polygon[:,0],polygon[:,1])
    ymin, ymax = min(polygon[:,1]),max(polygon[:,1])
    mmin, mmax = ymin,ymax
    attempt = ymin + (ymax-ymin) * densityRatio 
    pgbelow = makePolygonBelow(polygon,attempt)
    # plt.plot(pgbelow[:,0],pgbelow[:,1],label='initial attempt')

    newArea = PolyArea(pgbelow[:,0],pgbelow[:,1])

    ratio = newArea/originalArea

    while abs(ratio-densityRatio) > epsilon:
        if ratio-densityRatio > 0:
            mmax = attempt
        else:
            mmin = attempt
        attempt = (mmin+mmax)/2
        pgbelow_ = makePolygonBelow(polygon,attempt)
        newArea_ = PolyArea(pgbelow_[:,0],pgbelow_[:,1])
        ratio = newArea_/originalArea
        # print(f'{newArea_:.2f}',ratio)

    return attempt

def hydrostasy(contour,densityRatio=.917):
    """Returns center of mass and center of buoyancy for an immersed polygon.

    Args:
        contour (np.array): coordinates of polygon.
        densityRatio (float, optional): density ratio of the two fluids. Defaults to .917, ratio of ice/fresh water.

    Returns:
        tuple: (center of mass, center of buoyancy)
    """    
    polygon = Polygon(contour)
    com = polygon.centroid
    del polygon
    immersedPG = makePolygonBelow(contour,findWL(contour,densityRatio=densityRatio))
    polygon = Polygon(immersedPG)
    cob = polygon.centroid
    return com,cob