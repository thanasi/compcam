#!/usr/bin/env python
#########################
## doHW3.py
## put it all together to finish HW3 for computational cameras
##
## A. Athanassiadis
## Sept 2013
##
#########################
from __future__ import division
from pylab import imsave
from scipy.io import loadmat
import numpy as np

## import my own lightfield class
## hosted at https://github.com/thanasi/pylifi
from pylifi.LightFields import RLightField

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

######################################################

## load up the Dragon and Bunnies so that I can double check that my python code works as expected
#print 'loading test data...'
#lfData = loadmat('./LightField4D.mat')['lightField']
#
#lf = RLightField(lfData)
#lf.set_aperture(5)  ## set a circular aperture of diameter 5 images
#lf.set_focus(.5)    ## focus around the dragon's tail by cumulatively shifting the images by .5
#im1 = lf.render()   ## get the rendered image
#
#sim = lf.dump(mode=1, spacing=3)    ## get a grid that shows the spatial samples for each camera
#aim = lf.dump(mode=2, spacing=1)    ## get a grid that shows the angular samples for each scene pixel
#
#imsave('./Part0a.png', im1/255)
#imsave('./Part0b.png', sim/255)
#imsave('./Part0c.png', aim/255)

######################################################

## load in data as a 5D array (dy,dx,dv,du,colors)
print 'loading our data...'
lfData = np.load('./LFData2.npy')
lf = RLightField(lfData)


### Part 1a ###
print 'part 1a...'
## set the aperture to use all of the acquired images
lf.set_aperture(30)

## focus at infinity
lf.set_focus(0)

im = lf.render()
imsave('./Part1a.png', im/255)

### Part 1b ###
## focus in back (see-thru)
print 'part 1b...'
lf.set_center(15,0)         ## set the center to the middle of the image line
lf.set_aperture(30, False)  ## set largest aperture
lf.set_focus(-11.5)         ## focus on the dice

im = lf.render()
imsave('./Part1b.png', im/255)

### Part 1c ###
## focus in front
print 'part 1c...'
lf.set_focus(-18)           ## shift focus to foreground

im = lf.render()
imsave('./Part1c.png', im/255)

### Part 3c ###
## varying depth of field
print 'part 3c...'
lf.set_focus(-9)
lf.set_aperture(31, False)  ## shallowest depth of field

im = lf.render()
imsave('./Part3c-1.png', im/255)

lf.set_aperture(6, False)   ## much larger depth of field

im = lf.render()
imsave('./Part3c-2.png', im/255)

### Part 3b ###
### custom PSF
print 'part 3e...'
lf.set_center(15,0)
lf.set_aperture(10,False)   ## open up aperture again
lf.set_focus(-11.5)
im = lf.render()

gaussian = lambda x,c,s: np.exp(-(x-c)**2 / (2*s**2))
x = np.arange(0,29, dtype=np.float32)

mask = gaussian(x,3,5) + gaussian(x,27,5)  ## create a bimodal PSF

lf.set_mask(mask)
im2 = lf.render()

## check the PSF using the FFT of the grayscale images
img = rgb2gray(im)
im2g = rgb2gray(im2)

f1 = np.abs(np.fft.fftshift(np.fft.fft2(img)))
f2 = np.abs(np.fft.fftshift(np.fft.fft2(im2g)))

## output nice images to compare the PSFs
im3 = np.zeros((im.shape[0], im.shape[1] + im2.shape[1] + 5,3))
im3[:,:im.shape[1],:] = im
im3[:,-im2.shape[1]:,:] = im2
imsave('./Part3e-1.png', im3/255)

im4 = np.zeros((f1.shape[0], f1.shape[1] + f2.shape[1] + 5))
im4[:,:f1.shape[1]] = f1
im4[:,-f2.shape[1]:] = f2

imsave('./Part3e-2.png', im4, vmax=5000)