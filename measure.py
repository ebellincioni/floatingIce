#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:20:15 2023

@author: Edoardo Bellincioni
"""

from skimage.filters import gaussian
from skimage.color import rgb2gray
from skimage.exposure import equalize_adapthist
from skimage.transform import rescale, rotate
from skimage.segmentation import flood
from skimage.util import invert
#from skimage.morphology import binary_closing, disk
from skimage.draw import disk
from skimage.measure import moments, find_contours
import rawpy
import numpy as np
from tqdm import tqdm
from time import sleep
from matplotlib import pyplot as plt
import pyefd

def convert_compress(images,filepath):
    reads = []
    codes = []
    for img in tqdm(images):
        # read
        read = rgb2gray(rawpy.imread(img).postprocess())
        code = img[-8:-4]
        reads.append(read)
        codes.append(code)
    # np.savez_compressed(filepath,ututu=reads) # codes[i]=reads[i] for i in range(len(codes))
    np.save(filepath,reads)



def analyseNEF_flood(img,boolPrint,rescaling = 1,cropping = slice(0,-1), small_object_thresh=90,Gsigma=3,prevImg = None,boolPlot=False):
    '''
    See pipeline in presentation for explanation of this function. 

    Parameters
    ----------
    img : NEF image
        raw image of an iceball.
    boolPrint : boolean
        do you want to print which image is being analysed?
    rescaling : float<1.
        fraction for the image to be rescaled, default 0.25
    cropping : slice
        second coordinates of the images to be kept, default 350:1225
    small_object_thresh : int
        area in pixel of the small-object threshold, default 90
    Gsigma : float
        sigma of the gaussian filter 
    Fseed_point : tuple
        seed point for the second flooding
    prevImg : numpy 2D array
        previous analysed image
    boolPlot : boolean
        whether to plot or not

    Returns
    -------
    count : int
        count of pixels that correspond to the iceball.
    flooded2 : np.array
        image resulting from the processes.
    '''
    # read
    read = rgb2gray(rawpy.imread(img).postprocess())
    # rescale 
    rescaled = rescale(read, rescaling, anti_aliasing=True)
    # invert
    rescaled = invert(rescaled)
    # crop
    crop = rescaled[:,cropping]
    # calculate two std for thresholding
    stdStripe = (np.std(crop[:,-50:])**2+np.std(crop[-50:,:])**2)**.5
    # print(stdStripe)
    # colour for filling
    colour = np.mean([np.mean(crop[:,-50:]),np.mean(crop[-50:,:])])

    # gauss
    gauss = gaussian(crop,sigma=Gsigma)
    if boolPlot:
        plt.imshow(gauss, cmap='Greys_r')
        plt.title('gauss')
        plt.grid()
        plt.show()
    # flood
    flooded1 = np.invert(flood(gauss,seed_point=(0,0),tolerance=stdStripe*10))
    if boolPlot:
        plt.imshow(flooded1, cmap='Greys_r')
        plt.title('first flooding')
        plt.grid()
        plt.show()
    M = moments(flooded1,order=1)
    # print(M)
    if np.isnan(M).any() or np.sum(M)==0.: # if there is a nan in M
        Fseed_point=(int(flooded1.shape[0]/2),int(flooded1.shape[1]/2))
        print('BROKEN MOMENTS')
    else:
        Fseed_point = (int(M[1, 0] / M[0, 0]), int(M[0, 1] / M[0, 0]))
    mask = flood(flooded1.astype(float),seed_point=Fseed_point,tolerance=0)
    area = np.sum(mask==1) # area in px of the previous image
    radius = np.sqrt(area/np.pi)
    M = moments(mask,order=1)
    center = (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])
    if boolPlot:
        plt.imshow(mask, cmap='Greys_r')
        plt.title('mask without circle')
        plt.scatter(Fseed_point[1],Fseed_point[0])
        plt.grid()
        plt.show()
    multiplier = 1.02 # how much to increase the radius.
    if radius*multiplier < min(mask.shape)/2:
        rr, cc = disk((center[0], center[1]), radius*multiplier, shape=mask.shape)
    else:
        radius = min(mask.shape) / 2 
        rr, cc = disk((center[0], center[1]), radius*multiplier, shape=mask.shape)
    mask[rr, cc] = 1
    if boolPlot:
        plt.imshow(mask, cmap='Greys_r')
        plt.title('mask with circle')
        plt.grid()
        plt.show()
        plt.imshow(np.where(mask,gauss,colour), cmap='Greys_r')
        plt.title('masked gauss')
        plt.grid()
        plt.show()

    # contrast
    cEnhance = equalize_adapthist(crop)
    cEnhance = np.where(mask,cEnhance,colour)  
    #cEnhance = gaussian(cEnhance,sigma=Gsigma)
    if boolPlot:
        plt.imshow(cEnhance, cmap='Greys_r')
        plt.title('cEnhance with coloured mask')
        plt.grid()
        plt.show()
    flooded1 = flood(cEnhance,seed_point=(0,0),tolerance=stdStripe*3)
    if boolPlot:
        plt.imshow(flooded1, cmap='Greys_r')
        plt.title('flooded once')
        plt.grid()
        plt.show()
    flooded2 = flood(flooded1.astype(float),seed_point=Fseed_point,tolerance=0)
    
    # # flood
    sphereImg = flooded2
    # if boolPlot:
    #     plt.imshow(sphereImg, cmap='Greys_r')
    #     plt.title('flooded twice')
    #     plt.grid()
    #     plt.show()
    # count
    count = np.sum(sphereImg==1)
    # print
    if boolPrint:print(f'Image {img[-8:-4]} analysed')

        
    return count,sphereImg

def angleDependency(img,order,output_points):
    '''
    Returns the dependency of the melting rate on the angle

    Parameters
    ----------
    img : numpy 2D boolean array
        binarised image of the sphere
    order : int
        how many Fourier descriptor you want to use to fit the contour
    output_points : int
        how many points you want your Fourier-fitted contour to have

    Returns
    -------
    r_contour : numpy 2D array
        A list of x,y coordinates for the reconstructed contour.
    '''
    # find contours and select the longest one
    contours = find_contours(img)
    max_length = 0
    for i in range(len(contours)):
        length = len(contours[i])
        if length > len(contours[max_length]):
            max_length = i
    contour =  contours[max_length]

    # find fourier coefficients
    coeffs = pyefd.elliptic_fourier_descriptors(contour,
                                                order=order)
    # find center of transformation
    a0, c0 = pyefd.calculate_dc_coefficients(contour)
    # reconstruct contour
    r_contour = pyefd.reconstruct_contour(coeffs, locus=(a0,c0), 
                                          num_points=output_points)
    return r_contour
    


def removeHolder(img):
    '''
    Removes the holder from the sphere's binarised image.
    
    Parameters
    ----------
    img : numpy 2D boolean array
        binarised image of the sphere with the holder

    Returns
    -------
    sphere : numpy 2D boolean array
        binarised image of the sphere without the holder
    '''
    angle = np.rad2deg(np.arctan(1/3))
    sphereImg = rotate(img,angle,mode='constant',cval=0)

    grad = np.gradient(np.sum(sphereImg,axis=1))
    # running mean to smooth the gradient
    N=16
    grad = np.convolve(grad, np.ones(N)/N, mode='valid')
    holderLimit = np.where(grad==max(grad))[0][0]
    sphereImg[:holderLimit,:] = 0
    sphereImg = rotate(sphereImg,-angle,mode='constant',cval=0)
    
    return sphereImg

def loop(imagesArray,cropping, Gsigma):
    areas = []
    for index,img in enumerate(tqdm(imagesArray[::])):
        if index == 0:
            area, pImg = analyseNEF_flood(img,boolPrint=0,cropping=cropping,Gsigma=Gsigma)
        else:
            area, currImg = analyseNEF_flood(img,boolPrint=0,cropping=cropping,Gsigma=Gsigma,
                                             prevImg=None)
            # plt.imshow(currImg, cmap='Greys_r')
            # plt.grid()
            # plt.show()
            pImg = currImg.copy()
        areas.append(area)
       
    return areas

def cart2pol(x,y,center):
    '''
    Function that simply converts cartesian coordinates to polar coordinates. 

    Parameters
    ----------
    x : float
        x coordinate.
    y : float
        y coordinate.
    center : [float,float]
        coordinates of the center of the transformation in cartesian coord.

    Returns
    -------
    phi : float
        angle of polar coordinates.
    rho : float
        radius of polar coordinates.

    '''
    rho = np.sqrt((x-center[1])**2+(y-center[0])**2)
    phi = np.arctan2((y-center[0]),(x-center[1]))
    return phi,rho