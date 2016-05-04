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

import Encoding

import reedsolo
MULTI = False


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
    speed = 1200
    diff = 500
    if type(bits) is bitarray.bitarray:
        bits = np.unpackbits(bits)
    upsample = lcm((speed, fs))
    ratio = upsample/speed
    newBits = (np.repeat(bits, ratio).astype('float')*2)-1
    
    t = np.r_[0.0:len(newBits)-1]/(upsample)
    temp = np.cos(2*np.pi*t*1700-2*np.pi*diff*integrate.cumtrapz(newBits, dx=1.0/upsample))
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
    
def mafsk1200(bits, fs = 48000, baud = 1200, fd=1000, fc=2700):
    # the function will take a bitarray of bits and will output an AFSK1200 modulated signal of them, sampled at fs
    #  Inputs:
    #         bits  - bitarray of bits
    #         fs    - sampling rate
    # Outputs:
    #         sig    -  returns afsk1200 modulated signal
    
    # your code below:
    ck = lcm((baud, fs))
    factor = ck/fs
    bitlen = int(ck//baud)
    numbits =len(bits)
    
    arr = []
    hold = ''
    for x in bits:
        if x:
            hold += '1'
        else:
            hold += '0'
        if len(hold) == 2:
            if hold == '00':
                arr += [-2]*bitlen
            if hold == '01':
                arr += [-1]*bitlen
            if hold == '10':
                arr += [1]*bitlen
            if hold == '11':
                arr += [2]*bitlen
            hold = ''
    bit2 = np.array(arr)
    #bit2 = np.array(map(lambda x:[1]*bitlen if x else [-1]*bitlen, bits))
    #bit2 = np.reshape(bit2, (1, bitlen*numbits))[0]
    bit2.flatten()    #print len(m)
    #print r
    #return r
    m = bit2#np.array(m)
    t = r_[0:(numbits//baud+1)*ck]/ck
    r = fd*integrate.cumtrapz(m, dx = 1/ck)
    l = fc*t[:len(r)]
    w = l-r
    sig = np.cos(2*np.pi*w)
    
    sig = sig[::factor]
    return sig
                     

def nc_mafsk1200Demod(sig, fs=48000.0, baud=1200, TBW=2.0, fc = 2700, fd = 1000):
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
    
    N = (int(fs/baud*TBW)//2)*2+1
    f11 = fc - 2*fd
    f10 = fc - fd
    f01 = fc + fd
    f00 = fc + 2*fd
    
    # your code here
    taps = TBW*fs/1200-1
    taps = N
    filt = signal.firwin(taps, baud/2, window='hanning', nyq=fs/2)
    #plt.plot(np.fft.fft(filt))
    #plt.plot(filt)
    #f1 = 1200
    #f2 = 2200
    t2 = (r_[0:fs]/fs)[:taps]
    filt11 = filt* np.exp(t2*1j*f11*-2*np.pi)
    filt10 = filt* np.exp(t2*1j*f10*-2*np.pi)
    sig11 = signal.fftconvolve(sig, filt11, mode="same")
    sig10 = signal.fftconvolve(sig, filt10, mode="same")
    filt01 = filt* np.exp(t2*1j*f01*-2*np.pi)
    filt00 = filt* np.exp(t2*1j*f00*-2*np.pi)
    sig01 = signal.fftconvolve(sig, filt01, mode="same")
    sig00 = signal.fftconvolve(sig, filt00, mode="same")
    midsig = 0#(max(sig00)+min(sig00)+max(sig11)+min(sig11))/4
    
    sig11r = sig11 - midsig
    sig10r = sig10 - midsig
    sig01r = sig01 - midsig
    sig00r = sig00 - midsig
    
    return sig11r, sig10r, sig01r, sig00r
    
    diff = np.abs(sig12k)-np.abs(sig22k)
    return diff
    opt = signal.firwin(taps, baud*1.2, window='hanning', nyq=fs/2)
    ana = signal.fftconvolve(diff, opt, mode="same")
    #sign = np.sign(ana)

    NRZ = ana
    return NRZ

def bits2mag(bits, bitlen = 1):
    arr = []
    hold = ''
    for x in bits:
        if x:
            hold += '1'
        else:
            hold += '0'
        if len(hold) == 2:
            if hold == '00':
                arr += [-2]*bitlen
            if hold == '01':
                arr += [-1]*bitlen
            if hold == '10':
                arr += [1]*bitlen
            if hold == '11':
                arr += [2]*bitlen
            hold = ''
    return arr

def mafsk2barr(gg, centers=20,spacing=40,indii = None):
    x = np.vstack((np.abs(gg[0]), np.abs(gg[1]), np.abs(gg[2]), np.abs(gg[3])))
    x.shape
    if indii:
        o = np.argmax(x, axis=0)[indii]
    else:
        o = np.argmax(x,axis=0)[centers::spacing]
    y = []#bitarray.bitarray()
    for g in o:
        if g == 0:
            y.append(1)
            y.append(1)
        if g == 1:
            y.append(1)
            y.append(0)
        if g == 2:
            y.append(0)
            y.append(1)
        if g == 3:
            y.append(0)
            y.append(0)
    return np.array(y)
def mafsk2crossings(gg):
    x = np.vstack((np.abs(gg[0]), np.abs(gg[1]), np.abs(gg[2]), np.abs(gg[3])))
    #x.shape
    o = np.argmax(x, axis=0)
    #y = bitarray.bitarray()
    last = None
    zc = []
    for i, g in enumerate(o):
        if last != None:
            if last != g:
                zc.append(1)
            else:
                zc.append(0)
        else:
            zc.append(0)
        last = g
    return zc

def mPLL(NRZa, a = 0.74 , fs = 48000, baud = 1200):
    # 
    # function implements a simple phased lock loop for tyming recovery
    #
    # Inputs:
    #          NRZa -   The crossings signal
    #          a - nudge factor
    #          fs - sampling rate (arbitrary)
    #          baude  - the bit rate
    #
    # Outputs:
    #          idx - array of indexes to sample at
    #
    
    
    # Your code here
    idx = []
    inc = np.int32(2**32/(fs/baud))
    counter = np.int32(0)
    last = None
    for i, x in enumerate(NRZa):
        if x: 
            counter = counter*a
            counter = np.int32(counter)
        counter2 = counter+inc
        if counter2<counter:
            idx.append(i)
        last = x
        counter = counter2
    return idx




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
  if MULTI:
      dd = nc_mafsk1200Demod(data)
      plloc = mafsk2crossings(dd)
      bindex = mPLL(plloc)
      locs = mafsk2barr(dd, indii = bindex)
      samples = bitarray.bitarray([True if x == 1 else False for x in locs])
  else:
      demod = nc_afsk1200Demod(data, fs_usb)
      idx = PLL(demod, fs=fs_usb)
      samples = bitarray.bitarray([bit >= 0 for bit in np.array(demod)[idx]])
  bits = NRZI2NRZ(samples)
  packets = findPackets(bits)
  decoded = reduce(operator.add, packets)
  print decoded == eBits
  return decoded



def findPackets(bits, rs):
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
              if len(data[:-8]) > 16:
                done = True
            data = data[:-8]
            data = ax25.bit_unstuff(data)
            try:
              # print "received", data
              data = bitarray.bitarray(np.unpackbits(rs.decode(bytearray(bitarray.bitarray(data.to01()).tobytes()))).tolist())
              # print "decoded ", data
              if len(data) > 8 and checksum(data[:-8]) == data[-8:]:
                packets.append(data[:-8])
            except:
              pass
            #   print "error"
            #   print data
      else:
        b = bitstream.next()

    except StopIteration:
      break
  return packets
                

def packetize(bitstream, rs):
  """Converts bitstream to a list of packets following ax.25 protocol
  """
  infoSize = 8*220
  flags = bitarray.bitarray(np.tile([0,1,1,1,1,1,1,0],(3,)).tolist())
  b = bitstream
  packets = []
  while len(b) > 0:
    bits = b[:infoSize]
    bits += checksum(bits)
    # print "original", bits
    bits = bitarray.bitarray(np.unpackbits(rs.encode(bytearray(bits.tobytes()))).tolist())
    # print "ecced   ", bits
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


def transmit(bits, dusb_out):

  s = serial.Serial(port='COM3')
  s.setDTR(0)

  Qout = Queue.Queue()
  cQout = Queue.Queue()
  p = pyaudio.PyAudio()

  fs_usb = 48e3

  time.sleep(1)

  for packet in packetize(bits, rs=reedsolo.RSCodec(30)):
    Qout.put("KEYON")
    if MULTI:
        sig = mafsk1200(packet)
    else:
        sig = afsk1200(packet)
    Qout.put(sig*.3, fs_usb)
    Qout.put("KEYOFF")
    Qout.put(np.zeros(fs_usb//4))
  Qout.put("EOT")

  play_audio(Qout, cQout, p, fs_usb, dusb_out, s,0.2)

  while not(Qout.empty()) :
      time.sleep(1)

  time.sleep(1)

  p.terminate()
  s.close()


