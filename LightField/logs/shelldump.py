# coding: utf-8
import pylifi.LightFields as plf
import os
from pylab import *
from numpy import *

files = os.listdir('./')
ims = []
for f in files:
    ims.append(imread(f))
    
ims = np.array(ims,dtype=np.float32)
ims = ims[:,np.newaxis,:,:,:]
ims.shape
lf = plf.RLightField(ims2)
lf = plf.RLightField(ims)
im = lf.render()
imshow(im)
im = ims.mean(0,1)
im = ims.mean((0,1))
imshow(im)
ims.dtype
ims.max()
imshow(im1+im2)
imshow(ims.mean(0))
imshow(ims.mean(0).mean(0))
clf()
imshow(ims.sum(0).sum(0))
ims2 = ims[:10]
clf
clf()
lf = plf.RLightField(ims)
lf = plf.RLightField(ims2)
im = lf.render()
imshow(im)
im.min()
im.max()
lf.get_params
lf.get_params()
ims = ims.transpose(0,1)
ims = ims.transpose((0,1))
ims.transpose((0,1))
ims.shape
ims = np.transpose(ims,(0,1))
ims = np.transpose(ims,1)
ims.shape
ims.reshape(1,31,979,734,3)
ims = ims.reshape(1,31,979,734,3)
ims2 = ims[:,:10]
ims2.shape
lf = plf.RLightField(ims2)
im = lf.render()
imshow(im)
clf()
imshow(im)
im.max()
ims2 = ims[:,:5]
lf = plf.RLightField(ims2)
im = lf.render()
imshow(im)
clf()
imshow(im)
im/=255
imshow(im)
ims /= 255
imshow(ims[0])
imshow(ims[0,0])
clf()
imshow(ims[0,0])
lf = plf.RLightField(ims)
im = lf.render()
imshow(im)
imshow(ims[0,0])
lf.set_aperture(100)
lf.get_params()
lf.set_mask(np.ones(31))
im = lf.render()
figure()
imshow(im)
mask = abs(range(17-31))<5
mask = abs(range(17-31,17))<5
mask = abs(arange(17-31,17))<5
figure()
plot(mask)
close()
lf.set_mask(mask)
im2 = lf.render()
figure()
imshow(im2)
lf.set_focus(-3)
im2 = lf.render()
figure()
imshow(im2)
lf.set_focus(-5)
im2 = lf.render()
imshow(im2)
lf.set_focus(-8)
figure(3)
im2 = lf.render()
imshow(im2)
ims.shape
get_ipython().system(u'ls -F -G ')
save('../../LFData.npy',ims)
lf.data.shape
lf = plf.RLightField(ims[7:17])
im2 = lf.render()
lf = plf.RLightField(ims[:,7:17])
im2 = lf.render()
figure()
imshow(im2)
lf.set_mask(ones(10))
im2 = lf.render()
imshow(im2)
get_ipython().magic(u'pinfo np.roll')
from pylifi.MatchFeats import get_dxdy
for im1,im2 in np.vstack(ims[7:16],ims[8:17]):
    print get_dxdy(im1,im2)
    
for im1,im2 in np.vstack((ims[7:16],ims[8:17])):
    print get_dxdy(im1,im2)
    
for im1,im2 in np.hstack((ims[7:16],ims[8:17])):
    print get_dxdy(im1,im2)
    
for im1,im2 in np.vstack((ims[:,7:16],ims[:,8:17])):
    print get_dxdy(im1,im2)
    
for im1,im2 in np.hstack((ims[:,7:16],ims[:,8:17])):
    print get_dxdy(im1,im2)
    
np.hstack(ims[:,7:16],ims[:,8:17])
np.hstack((ims[:,7:16],ims[:,8:17]))
np.hstack((ims[:,7:16],ims[:,8:17])).shape
np.hstack((ims[:,7:16],ims[:,8:17]), axis=0).shape
np.hstack((ims[newaxis,:,7:16],ims[np.newaxis,:,8:17]).shape



)
np.hstack((ims[newaxis,:,7:16],ims[np.newaxis,:,8:17])).shape
for im1,im2 in np.hstack((ims[newaxis,:,7:16],ims[np.newaxis,:,8:17])).shape:
    print get_dxdy(im1,im2)
    
for im1,im2 in np.vstack((ims[newaxis,:,7:16],ims[np.newaxis,:,8:17])).shape:
    print get_dxdy(im1,im2)
    
for im1,im2 in np.vstack((ims[newaxis,:,7:16],ims[np.newaxis,:,8:17])):
    print get_dxdy(im1,im2)
    
for im1,im2 in np.hstack((ims[newaxis,:,7:16],ims[np.newaxis,:,8:17])):
    print get_dxdy(im1,im2)
    
import cv2
for im1,im2 in np.hstack((ims[newaxis,:,7:16],ims[np.newaxis,:,8:17])):
    print get_dxdy(cv2.cvtColor(im1,cv2.COLOR_RGB2GRAY),cv2.cvtColor(im2,cv2.COLOR_RGB2GRAY))
    
im1 = ims[7]
im1 = ims[:,7]
im2 = ims[:,8]
imshow(im1-im2)
ims1.shape
im1.shape
imshow(im1[0]-im2[0])
im1.dtype
ims /= 255
im1 = ims[:,7]
im2 = ims[:,8]
imshow(im1[0]-im2[0])
ims.dtype
imshow(ims[5])
imshow(ims[:,5])
imshow(ims[0,5,:,:,:])
ims[0,5,:,:,:].ptp()
ims[0,5,:,:,:].min()
ims *= 255
imshow(ims[0,5,:,:,:])
figure()
imshow(im1[0]-im2[0])
figure(); hist(im1[0]-im2[0])
figure(); hist((im1[0]-im2[0]).flat)
figure(); hist((im1[0]-im2[0]).flat, bins=30)
figure(5)
imshow((im1[0]-im2[0])>1)
imshow((im1[0]-im2[0])>.5)
imshow((im1[0]-im2[0])>.25)
imshow((im1[0]-im2[0])>.2)
imshow((im1[0]-im2[0])>.1)
imshow((im1[0]-im2[0])>.01)
imshow((im1[0]-im2[0])>.05)
im1 = cv2.cvtColor(ims[:,7],cv2.COLOR_RGB2GRAY)
im1 = cv2.cvtColor(ims[0,7],cv2.COLOR_RGB2GRAY)
im2 = cv2.cvtColor(ims[0,8],cv2.COLOR_RGB2GRAY)
figure)(
figure()
imshow(im1)
imshow(im1-im2)
get_dxdy(im1,im2)
im1.shape
im1=im1[:,:,newaxis]
im1.shape
im2=im2[:,:,newaxis]
get_dxdy(im1,im2)
im1 *= 255
im2 *= 255
get_dxdy(im1,im2)
im1.min()
im1.max()
im1 = im1.astype(uint8)
get_dxdy(im1,im2)
im2 = im2.astype(np.uint8)
im2.dtype
type(im2.dtype)
im2.dtype is np.uint8
im2.dtype is np.dtype("uint8")
get_dxdy(im1,im2)
im1 = cv2.cvtColor(ims[0,1],cv2.COLOR_RGB2GRAY)
im2 = cv2.cvtColor(ims[0,2],cv2.COLOR_RGB2GRAY)
im1.dtype
im1.max()
im1 *= 255
im2 *= 255
im1 = im1.astype(uint8)
im2 = im2.astype(uint8)
get_dxdy(im1,im2)
n = 3
im1 = (cv2.cvtColor(ims[0,n],cv2.COLOR_RGB2GRAY)*255).astype(np.uint8)
im2 = (cv2.cvtColor(ims[0,n+1],cv2.COLOR_RGB2GRAY)*255).astype(np.uint8)
for n in range(30):
    im1 = (cv2.cvtColor(ims[0,n],cv2.COLOR_RGB2GRAY)*255).astype(np.uint8)
    im2 = (cv2.cvtColor(ims[0,n+1],cv2.COLOR_RGB2GRAY)*255).astype(np.uint8)
    print get_dxdy(im1,im2)
    
figure()
lf.set_focus(-10)
im = lf.render()
imshow(im)
lf.set_focus(-9.5)
im = lf.render()
imshow(im)
lf.get_mask()
lf.set_focus(-9)
im = lf.render()
imshow(im)
lf.set_focus(-11); im = lf.render()
imshow(im)
lf = plf.RLightField(ims)
lf.set_focus(-11); im = lf.render()
imshow(im)
reload(plf)
lf = plf.RLightField(ims)
get_ipython().magic(u'pinfo lf.set_center')
lf.set_center(17,0)
lf.set_aperture(10)
lf.set_focus(-11)
im = lf.render()
figure()
imshow(im)
lf.set_aperture(10,False)
im = lf.render()
imshow(im)
lf.set_aperture(18,False)
lf.get_mask()
im = lf.render()
imshow(im)
lf.get_mask().shape
lf.get_mask().sum()
lf.get_params()
lf.set_aperture(31)
lf.get_mask()
lf.set_aperture(31,False)
lf.get_mask()
lf.get_center()
lf.acx
lf.acy
lf.set_aperture(35,False)
lf.get_mask.sum()
lf.get_mask().sum()
im = lf.render()
imshow(im)
lf.set_focus(-10.7)
im = lf.render()
figure(8)
imshow(im)
lf.set_focus(-11.5)
mask[:2] = 0
mask = lf.get_mask()
mask[:2] = 0
lf.set_mask(mask)
im = lf.render()
imshow(im)
im.min()
im = lf.render()
lf.get_mask
lf.get_mask()
lf.data
mask = ones(31)
mask[:2] = 0
mask = mask[newaxis,:]
mask
lf.set_mask(mask)
im = lf.render()
figure(8)
imshow(im)
get_ipython().system(u'ls -F -G ')
get_ipython().magic(u'cd ..')
get_ipython().system(u'ls -F -G ')
get_ipython().magic(u'cd ..')
get_ipython().system(u'ls -F -G ')
imsave('im17.seethru.png',im)
imshow(ims[0,17])
imshow(im)
imshow(ims[0,17])
imsave('im17.blocked.png',ims[0,17])
get_ipython().system(u'open .')
lf.set_focus(-16)
im = lf.render()
figure()
imshow(im)
lf.set_focus(-18)
imsave('im17.foc1.png',im)
im = lf.render()
figure()
imshow(im)
lf.set_focus(-19)
im = lf.render()
imshow(im)
lf.set_focus(-18.4)
im = lf.render()
imshow(im)
import scipy.ndimage as ndi
imshow(ndi.median_filter(im,10))
im2 = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
imshow(im2)
imshow(ndi.median_filter(im2,10))
imshow(ndi.median_filter(im2,15))
imshow(ndi.median_filter(im2,5))
imshow(ndi.median_filter(im2,8))
imshow(im)
imsave('im17.foc2.png',im)
lf.set_focus(-8)
im = lf.render()
figure()
imshow(im)
lf.set_focus(-6)
im = lf.render()
imshow(im)
lf.set_focus(-9)
im = lf.render()
imshow(im)
lf.set_aperture(10,False)
lf.set_focus(-9)
im = lf.render()
imshow(im)
lf.set_focus(-9.25)
im = lf.render()
imshow(im)
lf.set_focus(-9.5)
im = lf.render()
imshow(im)
lf.set_focus(-9.75)
im = lf.render()
imshow(im)
lf.set_focus(-10)
im = lf.render()
imshow(im)
lf.set_focus(-9.75)
im = lf.render()
imshow(im)
imsave('im17.salt.png',im)
lf.set_focus(-11)
im = lf.render()
imshow(im)
clf()
close('all')
lf.set_focus(-9.75)
im = lf.render()
lf.set_focus(-18)
im2 = lf.render()
imshow((im+im2)/2)
imshow((roll(im,8,axis=1)+im2)/2)
imshow((roll(im,7,axis=1)+im2)/2)
imshow((roll(im,5,axis=1)+im2)/2)
imshow((roll(im,-8,axis=1)+im2)/2)
imshow((roll(im,8,axis=1)+im2)/2)
lf.get_mask
lf.get_mask()
imA = lf.dump(1)
get_ipython().system(u'ls -F -G ')
imsave('dump1.png', imA)
imshow(imA)
imB = lf.dump(2)
imshow(imB)
imsave('dump2.png',imB)
