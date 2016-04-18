from __future__ import division
import numpy as np
from scipy import signal, ndimage, misc

class RiFi:

  def __init__(self, radio):
    self.radio = radio
    return

  def read(self, data):
    self.data = data
    self.original_dims = data.shape

  def send(self, data=None):
    if data is None:
      data = self.data

    # Downsample image
    self.data = data
    self.downsampled = self.downsample()
    self.preprocessed = self.preprocess()
    self.encoded = self.encode() # possibly unnecessary, transform into packets, etc.
    self.radio.transmit(self.encoded)
    return

  def receive(self, data):
    self.received = data
    self.decoded = self.decode()
    self.postprocessed = self.postprocess()
    return self.postprocessed

  def downsample(self, data=None, targetsize=(256,256)):
    if data is None:
      data = self.data
    
    data = signal.resample(data, targetsize[0], axis=0)
    data = signal.resample(data, targetsize[1], axis=1)

    return data

  def upsample(self, data=None, targetsize=None):
    if data is None:
      data = self.data
    if targetsize is None:
      targetsize = self.original_dims
      
    data = signal.resample(data, targetsize[0], axis=0)
    data = signal.resample(data, targetsize[1], axis=1)

    return data

  def preprocess(self, data=None):
    if data is None:
      data = self.downsampled
    return data

  def encode(self, data=None):
    if data is None:
      data = self.preprocessed
    return data

  def decode(self, data=None):
    if data is None:
      data = self.received
    return data

  def postprocess(self, data=None):
    if data is None:
      data = self.decoded
    return data


class Radio:
  """Only used for simulation purposes"""
  def __init__(self, receiver):
    self.receiver = receiver

  def transmit(self, data):
    # Add in random drops/ corruptions
    self.receiver.receive(data)
    return 

def PSNR(original, received, maxValue = 256):
  if original is RiFi:
    original = original.data
  if received is RiFi:
    received = received.postprocessed
  error2 = np.square(original - received)
  return 10 * np.log(maxValue**2 * np.prod(error2.shape) / np.sum(error2))

R = RiFi(None)
img = ndimage.imread('bird.jpg')
R.read(img)
ds = R.downsample(targetsize=(250, 350))
misc.imsave('bird_ds.jpg', ds)

rs = R.upsample(data=ds)
misc.imsave('bird_rs.jpg', rs)

# Sample Usage
# transmitter = RiFi()
# receiver = RiFi()
# radio = Radio(receiver)
# data = np.random.randint(256, size=(1,1024))
# transmitter.transmit(data)
# difference = transmitter.data - receiver.postprocessed
# print difference
# print "PSNR: ", PSNR(transmitter.data, receiver) # not a typo
