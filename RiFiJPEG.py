#!/usr/bin/env python
from __future__ import division
import numpy as np
from numpy import fft
from scipy import signal, fftpack, misc, ndimage
import bitarray
import sys

def RGB2YCbCr(RGB):
    """ Tranforms an NxMx3 RGB image array into a NxMx3 YCbCr image array.
        Both colorspaces should have a range of values from 0-255.
    """
    A = np.array([[[ 0.299,  0.587,  0.114],
                   [-0.169, -0.331,  0.500],
                   [ 0.500, -0.419, -0.081]]])
    
    YCbCr = np.dot(A, RGB.reshape((-1, 3)).T).T.reshape(RGB.shape) + np.array([0, 128, 128]).reshape((1,1,3))
    return YCbCr[:,:,0], YCbCr[:,:,1], YCbCr[:,:,2]

def YCbCr2RGB(y, cb, cr):
    """ Tranforms 3 NxM arrays corresponding to Y,Cb,Cr channels into a NxMx3 RGB image array.
        Both colorspaces should have a range of values from 0-255.
    """
    B = np.array([[1.000,  0.000,  1.400],
                  [1.000, -0.343, -0.711],
                  [1.000,  1.765,  0.000]])
    
    YCbCr = np.dstack((y, cb-128, cr-128))
    return np.dot(B, YCbCr.reshape((-1, 3)).T).T.reshape(YCbCr.shape)

def genZigZag(n):
    """ Generator used to calculate the "zigzag" traversal used to flatten JPEG blocks.
    """
    # Ascending
    swap = 0
    for idxSum in range(n):
        x = 0 if swap else idxSum
        y = idxSum if swap else 0
        if swap:
            while y >= 0:
                yield (x, y)
                x, y = x+1, y-1
        else:
            while x >= 0:
                yield (x, y)
                x, y = x-1, y+1
        swap ^= 1
    
    #Decending
    floor = 1
    for idxSum in range(n, 2*n-1):
        x = floor if swap else n-1
        y = n-1 if swap else floor
        if swap:
            while y >= floor:
                yield (x, y)
                x, y = x+1, y-1
        else:
            while x >= floor:
                yield (x, y)
                x, y = x-1, y+1
        swap ^= 1
        floor += 1

def DCT_2D(x):
    """ Performs the normalized 2D Discrete Cosine Transform
    >>> n = ndimage.read('bird.jpg')[:,:,0]
    >>> nDCT = DCT_2D(n)
    >>> nRec = IDCT_2D(nDCT)    # n ~= nRec
    """
    return fftpack.dct(fftpack.dct(x, norm='ortho', axis=0), norm='ortho', axis=1)

def IDCT_2D(x):
    """ Performs the normalized 2D Inverse Discrete Cosine Transform
    >>> n = ndimage.read('bird.jpg')[:,:,0]
    >>> nDCT = DCT_2D(n)
    >>> nRec = IDCT_2D(nDCT)    # n ~= nRec
    """
    return fftpack.idct(fftpack.idct(x, norm='ortho', axis=0), norm='ortho', axis=1)

def rescale(img, minval=0, maxval=255):
    """ Rescales the values of an array-like to be between 0 and 255 if
        the values exceed either bound.
        For safe casting to smaller-ranged datatypes, be sure to rescale beforehand.
    """
    if np.min(img) < minval:
        img = img - np.min(img)
    if np.max(img) > maxval:
        img = np.divide(img, np.max(img)) * maxval
    return img

class JPEGlib(object):
    # Standard-specified Luminosity Table
    QL = np.array([[16, 11, 10, 16,  24,  40,  51,  61],
                   [12, 12, 14, 19,  26,  58,  60,  55],
                   [14, 13, 16, 24,  40,  57,  69,  56],
                   [14, 17, 22, 29,  51,  87,  80,  62],
                   [18, 22, 37, 56,  68, 109, 103,  77],
                   [24, 35, 55, 64,  81, 104, 113,  92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]])
    
    # Standard-specified Chrominance Table
    QC = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                   [18, 21, 26, 66, 99, 99, 99, 99],
                   [24, 26, 56, 99, 99, 99, 99, 99],
                   [47, 66, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99]])
    
    
    def __init__(self, raw, qtable, QF=50, dims=None, rescale=False):
        assert 1 <= QF < 100, "Quality Factor (QF) must be 1 <= QF < 100"
        if QF > 50:
            alpha = 2 - QF/50
        else:
            alpha = 50/QF
        
        self.raw = raw
        self.qtable = qtable*alpha
        self.dims = raw.shape if dims is None else dims
        self.zstride = list(genZigZag(8))
        self.linstride = list(range(64))
        self.rescale = rescale


class JPEG(JPEGlib):
    def __init__(self, raw, qtable, QF):
        super(JPEG, self).__init__(raw, qtable, QF)

    def process(self, inarr=None, dims=None):
        if inarr is None:
            inarr = self.raw
        if dims is None:
            dims = self.dims
        out = np.zeros(((dims[0]//8)*(dims[1]//8), 64))
        
        for i in xrange(0, dims[0], 8):
            for j in xrange(0, dims[1], 8):
                block = inarr[i:i+8, j:j+8]
                block = DCT_2D(block)
                block = np.round(np.divide(block, self.qtable))
                for k, z in zip(self.linstride, self.zstride):
                    out[(dims[1]*i//64)+(j//8), k] = block[z[0], z[1]]
        return out

class IJPEG(JPEGlib):
    def __init__(self, raw, qtable, QF, dims, rescale=True):
        super(IJPEG, self).__init__(raw, qtable, QF, dims, rescale)

    def process(self, inarr=None, dims=None):
        if inarr is None:
            inarr = self.raw
        if dims is None:
            dims = self.dims
        out = np.zeros(self.dims)
        block = np.empty((8,8))

        for i in xrange(0, dims[0], 8):
            for j in xrange(0, dims[1], 8):
                for k, z in zip(self.linstride, self.zstride):
                    block[z[0], z[1]] = inarr[(dims[1]*i//64)+(j//8), k]
                block = np.multiply(block, self.qtable)
                block = IDCT_2D(block)
                out[i:i+8, j:j+8] = block

        return rescale(out) if self.rescale else out

def RiFi_preprocess(img):
    # Calculating image shape to be even multiples of 8.
    targetsize = img.shape[0] + (8 - (img.shape[0] % 8)) % 8, \
                 img.shape[1] + (8 - (img.shape[1] % 8)) % 8
    subsize = targetsize[0]//2, targetsize[1]//2                 # subsampling by 2x2
    subsize = subsize[0] + (8 - (subsize[0] % 8)) % 8, \
              subsize[1] + (8 - (subsize[1] % 8)) % 8
    
    # RGB -> YCbCr and subsampling (is it better to 0 pad instead?)
    imgY, imgCb, imgCr = RGB2YCbCr(img)
    imgY = signal.resample(signal.resample(imgY, targetsize[0], axis=0), targetsize[1], axis=1)
    imgCb = signal.resample(signal.resample(imgCb, subsize[0], axis=0), subsize[1], axis=1)
    imgCr = signal.resample(signal.resample(imgCr, subsize[0], axis=0), subsize[1], axis=1)
    
    # JPEG DCT transformation, quatization, and compression (?)
    bitsY = JPEG(imgY, qtable=JPEGlib.QL, QF=95).process()
    bitsCb = JPEG(imgCb, qtable=JPEGlib.QC, QF=95).process()
    bitsCr = JPEG(imgCr, qtable=JPEGlib.QC, QF=95).process()
    return targetsize, subsize, bitsY, bitsCb, bitsCr
    
def RiFi_postprocess(bitsY, bitsCb, bitsCr, targetsize, subsize, originalsize):
    # Use JPEG class to convert bitarrays back to images
    imgY = IJPEG(bitsY, qtable=JPEGlib.QL, QF=95, dims=targetsize).process()
    imgCb = IJPEG(bitsCb, qtable=JPEGlib.QC, QF=95, dims=subsize).process()
    imgCr = IJPEG(bitsCr, qtable=JPEGlib.QC, QF=95, dims=subsize).process()
    
    # Reconstruct RGB image in original dimensions
    imgY = signal.resample(signal.resample(imgY, originalsize[0], axis=0), originalsize[1], axis=1)
    imgCb = signal.resample(signal.resample(imgCb, originalsize[0], axis=0), originalsize[1], axis=1)
    imgCr = signal.resample(signal.resample(imgCr, originalsize[0], axis=0), originalsize[1], axis=1)
    imgRecon = YCbCr2RGB(imgY, imgCb, imgCr)

    # Rescale image and cast to uint8
    imgRecon[:,:,0] = rescale(imgRecon[:,:,0])
    imgRecon[:,:,1] = rescale(imgRecon[:,:,1])
    imgRecon[:,:,2] = rescale(imgRecon[:,:,2])
    return imgRecon.astype(np.uint8)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Example Usage: RiFiJPEG.py Images/bird.jpg"
    # Load image
    img = ndimage.imread(sys.argv[1])
    
    # Process and Reconstruct using JPEG
    tsize, ssize, Y, Cb, Cr = RiFi_preprocess(img)
    imgRecon = RiFi_postprocess(Y, Cb, Cr, tsize, ssize, (img.shape[0], img.shape[1]))
    
    # Save result
    misc.imsave(sys.argv[1].split('.')[0]+'_reconstructed.jpg', imgRecon)
