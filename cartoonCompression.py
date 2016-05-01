from __future__ import division
import numpy as np
from scipy import signal, ndimage, misc
import zlib
import bitarray
image_path = "Images/"

img = ndimage.imread(image_path+'bird.jpg')
