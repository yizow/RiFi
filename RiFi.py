from __future__ import division
import numpy as np
from scipy import signal, ndimage

class RiFi:

  def __init__(self, radio):
    self.radio = radio
    return

  def read(self, data):
    self.data = data

  def send(self, data=self.data):
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

  def downsample(self, data=self.data, targetsize=(256,256)):
    pad_len = targetsize - data.shape%targetsize
    ds_ratio = (data.shape+pad_len)/targetsize

    # zeropad
    data = np.pad(data, ((0, pad_len[0]), (0, pad_len[1])), mode='edge')
    
    # downsample
    # w = ???
    data = signal.convolve2d(data, w, mode='same')

    # decimate
    data = data[::ds_ratio[0], ::ds_ratio[1]]
    return data

  def preprocess(self, data=self.downsampled):
    return data

  def encode(self, data=self.preprocessed):
    return data

  def decode(self, data=self.received):
    return data

  def postprocess(self, data=self.decoded):
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

# Sample Usage
transmitter = RiFi()
receiver = RiFi()
radio = Radio(receiver)
data = np.random.randint(256, size=(1,1024))
transmitter.transmit(data)
difference = transmitter.data - receiver.postprocessed
print difference
print "PSNR: ", PSNR(transmitter.data, receiver) # not a typo
