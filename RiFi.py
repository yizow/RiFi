from __future__ import division
import numpy as np
from scipy import signal, ndimage, misc
import bitarray
image_path = "Images/"

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
  
  def factordownsample(self, data=None, factor=5):
    if data == None:
      data = self.data
    y = data.shape[0]
    x = data.shape[1]
    return self.downsample(data, targetsize=(y//factor, x//factor))

  def im2bitarray(self, data=None):
    if data == None:
      data = self.data
    output = []
    for bits_ref in data.T:
      asbit = map(lambda x:format(x, "08b"), bits_ref)
      asbit = reduce(lambda x, y:x+y, asbit)
      output.append(bitarray.bitarray(asbit))
    return output

class Radio:"""Only used for simulation purposes"""
  """Only used for simulation purposes"""
  def __init__(self, receiver):
    self.receiver = receiver

  def transmit(self, data):
    # Add in random drops/ corruptions
    self.receiver.receive(data)
    return 

def PSNR(original, received, maxValue = 255):
  if original is RiFi:
    original = original.data
  if received is RiFi:
    received = received.postprocessed
  original = original.astype('float64')
  received = received.astype('float64')
  error2 = np.square(original - received)
  return 10 * np.log10(maxValue**2 * np.prod(error2.shape) / np.sum(error2))

R = RiFi(None)
img = ndimage.imread(image_path+'bird.jpg')
R.read(img)
#ds = R.downsample(targetsize=(250, 350))

#rs = R.upsample()

ds2 = R.factordownsample(factor = 5)
misc.imsave('bird_ds2.jpg', ds2)
rs = R.upsample(ds2)
misc.imsave('bird_rs2.jpg', rs)

misc.imsave(image_path+'bird_ds.jpg', ds)

misc.imsave(image_path+'bird_rs.jpg', rs)

# Sample Usage
# transmitter = RiFi()
# receiver = RiFi()
# radio = Radio(receiver)
# data = np.random.randint(256, size=(1,1024))
# transmitter.transmit(data)
# difference = transmitter.data - receiver.postprocessed
# print difference
# print "PSNR: ", PSNR(transmitter.data, receiver) # not a typo
