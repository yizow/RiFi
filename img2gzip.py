#%pylab
from __future__ import division
if True:
  # Import functions and libraries
  import numpy as np
  import matplotlib.pyplot as plt
  #import pyaudio
  import Queue
   
  import struct
  import gzip
  import threading,time
  import sys

  from numpy import pi
  from numpy import sin
  from numpy import zeros
  from numpy import r_
  from numpy import ones
  from scipy import signal
  from scipy import integrate
  from scipy import signal, ndimage, misc, fftpack

  import threading,time
  import multiprocessing

  from numpy import mean
  from numpy import power
  from numpy.fft import fft
  from numpy.fft import fftshift
  from numpy.fft import ifft
  from numpy.fft import ifftshift
  from  scipy.io.wavfile import read as wavread
  #import serial
  import ax25
  from fractions import gcd
  import bitarray

IMG_DIR = 'Images/'
tempDir = 'temp/'

def intarr2bitarr(w):
    bits_ref = map(lambda x:format(x, "08b"), w)
    bits_ref = reduce(lambda x, y:x+y, bits_ref)
    #print bits_ref
    bits_ref = bitarray.bitarray(bits_ref)
    return bits_ref

def bitarr2intarr(br):
    output = []
    working = ''
    for x in br:
        if x:
            working += '1'
        else:
            working += '0'
        if len(working) == 8:
            output.append(int(working, 2))
            working = '' 
    return output

def img2gzip(name, img = None, fmt = '.tiff'):
    if img == None:
        img = ndimage.imread(IMG_DIR+name+fmt)
    np.save(name, img)
    with open(tempDir+name+'.npy', 'rb') as f:
        w = f.read()
    with gzip.open(tempDir+name+'.gz', 'wb') as f:
        f.write(w)
    misc.imsave(tempDir+name+'2.jpg', img)

def gzip2img(name):
    with gzip.open(tempDir+name+'.gz', 'rb') as f:
        e = f.read() 
    with open(tempDir+name+'2.npy', 'w') as f:
        f.write(e)
    return np.load(name+'2.npy')

def gz2bitarr(name):
    with open(tempDir+name+'.gz', 'rb') as f:
        w = f.read() 
    asints = struct.unpack('B'*len(w), w)
    return intarr2bitarr(asints)

def bitarr2gz(name, arr):    
    w = bitarr2intarr(arr)
    y = struct.pack('B'*len(w), *w)
    with open(tempDir+name+'.gz', 'wb') as f:
        f.write(y)






name = 'pauly'
img = ndimage.imread(IMG_DIR+name+'.tiff')
img2gzip(name)
imgtest = gzip2img(name)
#plt.imshow(imgtest)
#plt.imshow(img)
statorig = os.stat(IMG_DIR+name+'.tiff').st_size
statcomp = os.stat(tempDir+name+'.gz').st_size
statjpg = os.stat(tempDir+name+'2.jpg').st_size
print "compression ratio: {}".format(statcomp/statorig)
print "jpg ratio: {}".format(statjpg/statorig)

#print imgtest[0][0][0]
#imgtest[0][0][0] -= 1
print "psnr",  PSNR(img, imgtest)

