import itertools
import numpy as np
import bitarray
import Queue

def encode(DCT, huffmanTableDC, huffmanTableAC):
  """Performs DCPM on DC components, and RLE on AC components
  Expects an N*64 matrix, where each row is a single MCU
  DCT[:,0] are the DC components
  Outputs a bitstream
  """
  DCValues = DCPM(DCT[:, 0] )
  DCBits = [encodeDC(value, huffmanTableDC) for value in DCValues]

  ACValues = DCT[:,1:]
  ACBits = [encodeAC(values, huffmanTableAC) for values in ACValues]
  bits = bitarray.bitarray()
  for i in range(len(DCBits)):
    bits += DCBits[i] + ACBits[i]
  bits = bitarray.bitarray(np.binary_repr(216, width=8)) + bits
  length = len(bits) 
  if length % 8 > 0:
    offset = 8 - length % 8
    # print "padding", offset
    bits = bitarray.bitarray([0]*offset) + bits
  bits += bitarray.bitarray("1111111111011001")
  return bits

def decode(bits, huffmanRootDC, huffmanRootAC):
  """Expects a bitstream, outputs an N*64 matrix
  """
  # bits = iter(bitstream)
  DCValues = np.array([])
  ACValues = []
  lastDC = 0

  startMarker = []
  for _ in range(8):
    startMarker.append(bits.next())
  startMarker = bitarray.bitarray(startMarker)
  for _ in range(8):
    if startMarker.to01() == "11011000":
      break
    startMarker.pop(0)
    startMarker.append(bits.next())
  while True:
    try: 
      testEnd = bitarray.bitarray()
      for _ in range(16):
        testEnd.append(bits.next())
      if testEnd.to01() == "1111111111011001":
        break
      bits = itertools.chain(testEnd, bits)

      # DC Value
      node = huffmanRootDC
      while node.code == None:
        if bits.next():
          node = node.right
        else:
          node = node.left
      code = bitarray.bitarray(np.binary_repr(node.code, width=8))
      size = int(code.to01(), 2)
      if size == 0:
        diffDC = 0
      else:
        value = bitarray.bitarray()
        for _ in range(size):
          value.append(bits.next())
        if value[0] == False:
          value.invert()
          diffDC = -1 * int(value.to01(), 2)
        else:
          diffDC = int(value.to01(), 2)
      # print code, size, value

      lastDC += diffDC
      DCValues = np.append(DCValues, lastDC)

      # AC Values
      values = np.zeros(63)
      index = 0
      while index < len(values):
        node = huffmanRootAC
        while node.code == None:
          if bits.next():
            node = node.right
          else:
            node = node.left
        code = bitarray.bitarray(np.binary_repr(node.code, width=8))
        zeroRun = int(code[:4].to01(), 2)
        size = int(code[4:].to01(), 2)
        # End Of Block
        if zeroRun == 0 and size == 0:
          break
        if zeroRun == 15 and size == 0:
          index += 16 
          continue
        index += zeroRun
        value = bitarray.bitarray()
        for _ in range(size):
          value.append(bits.next())
        # print code, size, zeroRun, value
        if value[0] == False:
          value.invert()
          value = -1 * int(value.to01(), 2)
        else:
          value = int(value.to01(), 2)
        values[index] = value
        index += 1
      ACValues.append(values)
    except StopIteration:
      break

  DCValues = DCValues.reshape((len(DCValues), 1))
  ACValues = np.array(ACValues)
  # print DCValues.shape, ACValues.shape
  return np.hstack((DCValues, ACValues))

def DCPM(values):
  pre = np.insert(np.array(values), 0, 0)
  post = np.append(np.array(values), 0)
  return (post - pre)[:-1]

def encodeDC(value, table):
  b = bitarray.bitarray()
  if value == 0:
    # print value, bitarray.bitarray(np.binary_repr(0, width=8))
    return bitarray.bitarray(table[0])
  size = int(np.log2(abs(value))) + 1
  b += bitarray.bitarray(table[size])
  c = bitarray.bitarray(np.binary_repr(abs(value)))
  if value < 0:
    c.invert()
  # print value, b, c
  return b + c

def encodeAC(values, table):
  lastIndex = np.nonzero(np.append(values, 1))[0][-1]
  zeroRun = 0
  bits = bitarray.bitarray()
  for i in range(len(values)):
    value = values[i]
    if i > lastIndex:
      bits += bitarray.bitarray(table[0])
      return bits
    if value == 0:
      if zeroRun == 15:
        bits += bitarray.bitarray(table[240])
        zeroRun = 0
        continue
      zeroRun += 1
      continue
    # bits += bitarray.bitarray(np.binary_repr(zeroRun, width=4))
    size = int(np.log2(abs(value))) + 1 
    # bits += bitarray.bitarray(np.binary_repr(size, width=4))
    bits += bitarray.bitarray(table[zeroRun * 16 + size])
    v = bitarray.bitarray(np.binary_repr(abs(value)))
    if value < 0:
      v.invert()
    bits += v
    zeroRun = 0
  return bits


tableLuminanceDC = {
  1: [],
  2: [0x00],
  3: [0x01, 0x02, 0x03, 0x04, 0x05],
  4: [0x06], 
  5: [0x07],
  6: [0x08], 
  7: [0x09],
  8: [0x0A],
  9: [0x0B],
  10: [0x0C],
}

tableChrominanceDC = {
  1: [],
  2: [0x00, 0x01, 0x02],
  3: [0x03],
  4: [0x04], 
  5: [0x05],
  6: [0x06], 
  7: [0x07],
  8: [0x08],
  9: [0x09],
  10: [0x0A],
  11: [0x0B],
  12: [0x0C],
}

def readACTable(bits):
  """bits is a hex string
  returns a table mapping bitlengths to codes"""
  table = {}
  length = bits[:4]
  tc = bits[4]
  th = bits[5]
  bits = bits[6:]
  codeCounts = [0]*16

  for i in range(16):
    codeCounts[i] = int(bits[:2], 16)
    bits = bits[2:]


  for i in range(16):
    numCodes = codeCounts[i]
    table[i+1] = []
    for _ in range(numCodes):
      table[i+1].append(int(bits[:2], 16))
      bits = bits[2:]
  return table

tableLuminanceAC = readACTable('00b5100002010303020403050504040000017d01020300041105122131410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9fa')

tableChrominanceAC = readACTable('00b51100020102040403040705040400010277000102031104052131061241510761711322328108144291a1b1c109233352f0156272d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758595a636465666768696a737475767778797a82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9fa')


class HuffmanNode():

  def __init__(self, parent=None, direction=0):
    if parent == None:
      self.bits = ""
    else: 
      self.bits = parent.bits + str(direction)
    self.left = None
    self.right = None
    self.code = None

  def __str__(self):
    return "bits, left, right, code:" + str((self.bits, self.left != None, self.right != None, self.code))


def createHuffmanTree(table):
  """table is a table of bit lengths and corresponding code words
  root is the root of the resulting huffman tree
  huffman is a dictionary mapping codes to bits
  """
  root = HuffmanNode()
  leaves = Queue.Queue()
  root.left = HuffmanNode(root, 0)
  root.right = HuffmanNode(root, 1)
  leaves.put(root.left)
  leaves.put(root.right)
  huffman = {}

  for key in range(1, len(table.keys())+1):
    for code in table[key]:
      leaf = leaves.get()
      leaf.code = code
      huffman[code] = leaf.bits

    nextLeaves = Queue.Queue()
    while not leaves.empty():
      node = leaves.get()
      node.left = HuffmanNode(node, 0)
      node.right = HuffmanNode(node, 1)
      nextLeaves.put(node.left)
      nextLeaves.put(node.right)
    leaves = nextLeaves

  return root, huffman


huffmanRootLuminanceDC, huffmanLookupLuminanceDC = createHuffmanTree(tableLuminanceDC)
huffmanRootChrominanceDC, huffmanLookupChrominanceDC = createHuffmanTree(tableChrominanceDC)

huffmanRootLuminanceAC, huffmanLookupLuminanceAC = createHuffmanTree(tableLuminanceAC)
huffmanRootChrominanceAC, huffmanLookupChrominanceAC = createHuffmanTree(tableChrominanceAC)