import numpy as np 
import pyaudio
import ax25
import bitarray
import serial
import Queue
import threading, time
import sys
from fractions import gcd

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from numpy import ones
from scipy import signal
from scipy import integrate

import operator

def printDevNumbers(p):
    N = p.get_device_count()
    for n in range(0,N):
        name = p.get_device_info_by_index(n).get('name')
        print n, name
                
def afsk1200(bits, fs = 48000):
    # the function will take a bitarray of bits and will output an AFSK1200 modulated signal of them, sampled at 44100Hz
    #  Inputs:
    #         bits  - bitarray of bits
    #         fs    - sampling rate
    # Outputs:
    #         sig    -  returns afsk1200 modulated signal
    # your code below:
    if type(bits) is bitarray.bitarray:
        bits = np.unpackbits(bits)
    upsample = lcm((1200, fs))
    ratio = upsample/1200
    newBits = (np.repeat(bits, ratio).astype('float')*2)-1
    
    t = np.r_[0.0:len(newBits)-1]/(upsample)
    temp = np.cos(2*np.pi*t*1700-2*np.pi*500*integrate.cumtrapz(newBits, dx=1.0/upsample))
    sig = temp[::upsample/fs]
    
    return sig


def nc_afsk1200Demod(sig, fs=48000.0, TBW=2.0):
    #  non-coherent demodulation of afsk1200
    # function returns the NRZ (without rectifying it)
    # 
    # sig  - signal
    # baud - The bitrate. Default 1200
    # fs   - sampling rate in Hz
    # TBW  - TBW product of the filters
    #
    # Returns:
    #     NRZ  
    # your code here
    taps = fs/600-1
    bandpass = signal.firwin(taps, 600, nyq=fs/2)
    spacepass = bandpass * np.exp(1j*2*np.pi*1200*np.r_[0.0:taps]/fs)
    markpass = bandpass * np.exp(1j*2*np.pi*2200*np.r_[0.0:taps]/fs)
    spaces = signal.fftconvolve(sig, spacepass, mode='same')
    marks = signal.fftconvolve(sig, markpass, mode='same')

    analog = np.abs(spaces)-np.abs(marks)
    lowpass = signal.firwin(taps, 1200*1.2, nyq=fs/2)
    filtered = signal.fftconvolve(analog, lowpass, mode='same')
    NRZ = filtered
    
    return NRZ


def PLL(NRZa, a = 0.74 , fs = 48000, baud = 1200):
    # 
    # function implements a simple phased lock loop for tyming recovery
    #
    # Inputs:
    #          NRZa -   The NRZ signal
    #          a - nudge factor
    #          fs - sampling rate (arbitrary)
    #          baude  - the bit rate
    #
    # Outputs:
    #          idx - array of indexes to sample at
    #
    # Your code here
    idx = []
    increment = 2**32 * baud / fs 
    counter = np.int32(increment)
    
    for i in range(1, len(NRZa)):
        crossing = np.sign(NRZa[i]) != np.sign(NRZa[i-1])
        if crossing:
            counter = int(a*counter)
        counter += increment
        if counter >= 2**31:
            counter = np.int32(counter)
            idx.append(i)

    return np.array(idx).astype('int32') 
    

def NRZ2NRZI(NRZ):
    
    NRZI = NRZ.copy() 
    current = True
    for n in range(0,len(NRZ)):
        if NRZ[n] :
            NRZI[n] = current
        else:
            NRZI[n] = not(current)
        current = NRZI[n]
    return NRZI

def NRZI2NRZ(NRZI, current = True):
    
    NRZ = NRZI.copy() 
    
    for n in range(0,len(NRZI)):
        NRZ[n] = NRZI[n] == current
        current = NRZI[n]
    
    return NRZ

# function to generate a checksum for validating packets
def genfcs(bits):
    # Generates a checksum from packet bits
    fcs = ax25.FCS()
    for bit in bits:
        fcs.update_bit(bit)
    
    digest = bitarray.bitarray(endian="little")
    digest.frombytes(fcs.digest())

    return digest

# function to parse packet bits to information
def decodeAX25(bits):
    ax = ax25.AX25()
    ax.info = "bad packet"
    
    
    bitsu = ax25.bit_unstuff(bits[8:-8])
    
    if (genfcs(bitsu[:-16]).tobytes() == bitsu[-16:].tobytes()) == False:
        #print("failed fcs")
        return ax
    
    bytes = bitsu.tobytes()
    ax.destination = ax.callsign_decode(bitsu[:56])
    source = ax.callsign_decode(bitsu[56:112])
    if source[-1].isdigit() and source[-1]!="0":
        ax.source = b"".join((source[:-1],'-',source[-1]))
    else:
        ax.source = source[:-1]
    
    digilen=0    
    
    if bytes[14]=='\x03' and bytes[15]=='\xf0':
        digilen = 0
    else:
        for n in range(14,len(bytes)-1):
            if ord(bytes[n]) & 1:
                digilen = (n-14)+1
                break

#    if digilen > 56:
#        return ax
    ax.digipeaters =  ax.callsign_decode(bitsu[112:112+digilen*8])
    ax.info = bitsu[112+digilen*8+16:-16].tobytes()
    
    return ax

def play_audio( Q,ctrlQ ,p, fs , dev, ser="", keydelay=0.1):
    # play_audio plays audio with sampling rate = fs
    # Q - A queue object from which to play
    # ctrlQ - A queue object for ending the thread
    # p   - pyAudio object
    # fs  - sampling rate
    # dev - device number
    # ser - pyserial device to key the radio
    # keydelay - delay after keying the radio
    #
    #
    # There are two ways to end the thread: 
    #    1 - send "EOT" through  the control queue. This is used to terminate the thread on demand
    #    2 - send "EOT" through the data queue. This is used to terminate the thread when data is done. 
    #
    # You can also key the radio either through the data queu and the control queue
    
    
    # open output stream
    ostream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),output=True,output_device_index=dev)
    # play audio
    while (1):
        if not ctrlQ.empty():
            
            # control queue 
            ctrlmd = ctrlQ.get()
            if ctrlmd is "EOT"  :
                    ostream.stop_stream()
                    ostream.close()
                    print("Closed  play thread")
                    return;
            elif (ctrlmd is "KEYOFF"  and ser!=""):
                ser.setDTR(0)
                #print("keyoff\n")
            elif (ctrlmd is "KEYON" and ser!=""):
                ser.setDTR(1)  # key PTT
                #print("keyon\n")
                time.sleep(keydelay) # wait 200ms (default) to let the power amp to ramp up
                
        
        data = Q.get()
        
        if (data is "EOT") :
            ostream.stop_stream()
            ostream.close()
            print("Closed  play thread")
            return;
        elif (data is "KEYOFF"  and ser!=""):
            ser.setDTR(0)
            #print("keyoff\n")
        elif (data is "KEYON" and ser!=""):
            ser.setDTR(1)  # key PTT
            #print("keyon\n")
            time.sleep(keydelay) # wait 200ms (default) to let the power amp to ramp up
            
        else:
            try:
                ostream.write( data.astype(np.float32).tostring() )
            except:
                print("Exception")
                break
            
def record_audio( queue,ctrlQ, p, fs ,dev,chunk=512):
    # record_audio records audio with sampling rate = fs
    # queue - output data queue
    # p     - pyAudio object
    # fs    - sampling rate
    # dev   - device number 
    # chunk - chunks of samples at a time default 1024
    #
    # Example:
    # fs = 44100
    # Q = Queue.queue()
    # p = pyaudio.PyAudio() #instantiate PyAudio
    # record_audio( Q, p, fs, 1) # 
    # p.terminate() # terminate pyAudio
    
   
    istream = p.open(format=pyaudio.paFloat32, channels=1, rate=int(fs),input=True,input_device_index=dev,frames_per_buffer=chunk)

    # record audio in chunks and append to frames
    frames = [];
    while (1):
        if not ctrlQ.empty():
            ctrlmd = ctrlQ.get()          
            if ctrlmd is "EOT"  :
                istream.stop_stream()
                istream.close()
                print("Closed  record thread")
                return;
        try:  # when the pyaudio object is distroyed stops
            data_str = istream.read(chunk) # read a chunk of data
        except:
            break
        data_flt = np.fromstring( data_str, 'float32' ) # convert string to float
        queue.put( data_flt ) # append to list


def lcm(numbers):
    return reduce(lambda x, y: (x*y)/gcd(x,y), numbers, 1)

def testTransmit(eBits, debug=False):
  if debug:
    p = pyaudio.PyAudio()
    printDevNumbers(p)
    p.terminate()

  dusb_in = 2
  dusb_out = 2
  din = 9
  dout = 9

  s = serial.Serial(port='/dev/ttyUSB0')
  s.setDTR(0)

  Qin = Queue.Queue()
  cQin = Queue.Queue()
  Qout = Queue.Queue()
  cQout = Queue.Queue()

  p = pyaudio.PyAudio()

  fs_usb = 48e3

  t_rec = threading.Thread(target = record_audio,   args = (Qin, cQin, p, fs_usb, dusb_in, 512))
  t_rec.start()
  time.sleep(1)

  for packet in packetize(eBits):
    Qout.put("KEYON")
    Qout.put(afsk1200(packet)*.1, fs_usb)
    Qout.put("KEYOFF")
    Qout.put(np.zeros(fs_usb//2))
  Qout.put("EOT")

  play_audio(Qout, cQout, p, fs_usb, dusb_out, s,0.2)

  while not(Qout.empty()) :
      time.sleep(1)


  time.sleep(1)
  cQin.put('EOT')
  time.sleep(1) # give time for the thread to get killed


  p.terminate()
  s.close()

  recorded = []
  while not (Qin.empty()):
      recorded.append(Qin.get())

  data = np.array(recorded).flatten()
  demod = nc_afsk1200Demod(data, fs_usb)
  idx = PLL(demod, fs=fs_usb)
  samples = bitarray.bitarray([bit >= 0 for bit in np.array(demod)[idx]])
  bits = NRZI2NRZ(samples)
  packets = findPackets(bits)
  decoded = reduce(operator.add, packets)
  print decoded == eBits
  return decoded



def findPackets(bits):
  if len(bits) == 0:
    return []
  flag = "01111110"
  bitstream = iter(bits)
  b = bitstream.next()
  packets = []
  while True:
    try:
      if b == 0:  
        for _ in range(6):
          b = bitstream.next()
          if b != 1:
            break
        else:
          b = bitstream.next()
          if b == 0:
            # Flag found begin collecting data 
            data = bitarray.bitarray()
            done = False
            while not done:
              for _ in range(8):
                data.append(bitstream.next())
              while data[-8:].to01() != flag:
                data.append(bitstream.next())
              if data[:8].to01() == flag:
                data = data[8:]
              if len(data[:-8]) > 0:
                done = True
            data = data[:-8]
            data = ax25.bit_unstuff(data)
            if len(data) > 8 and checksum(data[:-8]) == data[-8:]:
              packets.append(data[:-8])
      else:
        b = bitstream.next()

    except StopIteration:
      break
  return packets
                

def packetize(bitstream):
  """Converts bitstream to a list of packets following ax.25 protocol
  """
  infoSize = 8*256
  flags = bitarray.bitarray(np.tile([0,1,1,1,1,1,1,0],(3,)).tolist())
  b = bitstream
  packets = []
  while len(b) > 0:
    bits = b[:infoSize]
    bits += checksum(bits)
    b = b[infoSize:]
    padded = flags + bitarray.bitarray(ax25.bit_stuff(bits)) + flags
    packets.append(NRZ2NRZI(padded))
  return packets

def checksum(bits):
  power = 0
  total = 0
  for b in bits:
    if b:
      total += 2**power
    power += 1
    power %= 8
    total %= 256
  return bitarray.bitarray(np.binary_repr(total, width=8))


def findPacketBuffer(eBits):
  dusb_in = 2
  dusb_out = 2
  din = 9
  dout = 9

  output = []

  for divisor in [12]:
    correct = 0
    for _ in range(1):
      s = serial.Serial(port='/dev/ttyUSB0')
      s.setDTR(0)

      Qin = Queue.Queue()
      cQin = Queue.Queue()
      Qout = Queue.Queue()
      cQout = Queue.Queue()
      p = pyaudio.PyAudio()

      fs_usb = 48e3

      t_rec = threading.Thread(target = record_audio,   args = (Qin, cQin, p, fs_usb, dusb_in, 512))
      t_rec.start()
      time.sleep(1)

      for packet in packetize(eBits):
        Qout.put("KEYON")
        Qout.put(afsk1200(packet)*.3, fs_usb)
        Qout.put("KEYOFF")
        Qout.put(np.zeros(fs_usb//divisor))
      Qout.put("EOT")

      play_audio(Qout, cQout, p, fs_usb, dusb_out, s,0.2)

      while not(Qout.empty()) :
          time.sleep(1)


      time.sleep(1)
      cQin.put('EOT')
      time.sleep(2) # give time for the thread to get killed


      p.terminate()
      s.close()

      recorded = []
      while not (Qin.empty()):
          recorded.append(Qin.get())

      data = np.array(recorded).flatten()
      demod = nc_afsk1200Demod(data, fs_usb)
      idx = PLL(demod, fs=fs_usb)
      samples = bitarray.bitarray([bit >= 0 for bit in np.array(demod)[idx]])
      bits = NRZI2NRZ(samples)
      packets = findPackets(bits)
      decoded = reduce(operator.add, packets, bitarray.bitarray())
      correct += decoded == eBits
    output.append(correct*1.0/5)
  return output