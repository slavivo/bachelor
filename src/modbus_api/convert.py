import struct

def int16_to_uint16(val):
    ba = bytearray(struct.pack(">h", val)) 
    msb = struct.unpack(">H",ba[0:2])
    return msb[0]

def uint32_to_uint16(val):
    ba = bytearray(struct.pack(">L", val)) 
    msb = struct.unpack(">H",ba[0:2])
    lsb = struct.unpack(">H",ba[2:4])
    return msb[0],lsb[0]
    
def int32_to_uint16(val):
    ba = bytearray(struct.pack(">l", val)) 
    msb = struct.unpack(">H",ba[0:2])
    lsb = struct.unpack(">H",ba[2:4])
    return msb[0],lsb[0]

def float32_to_uint16(val):
    ba = bytearray(struct.pack(">f", val)) 
    msb = struct.unpack(">H",ba[0:2])
    lsb = struct.unpack(">H",ba[2:4])
    return msb[0],lsb[0]

def uint64_to_uint16(val):
    ba = bytearray(struct.pack(">Q", val)) 
    mmsb = struct.unpack(">H",ba[0:2])
    mlsb = struct.unpack(">H",ba[2:4])
    lmsb = struct.unpack(">H",ba[4:6])
    llsb = struct.unpack(">H",ba[6:8])
    return mmsb[0],mlsb[0],lmsb[0],llsb[0]

def uint16_to_int16(LSB):
    uint16_lsb = LSB.to_bytes(2, byteorder='big', signed=False)
    uint16 = struct.unpack(">h",uint16_lsb)
    return uint16[0]

def uint16_to_uint32(MSB,LSB):
    uint32_msb = MSB.to_bytes(2, byteorder='big', signed=False)
    uint32_lsb = LSB.to_bytes(2, byteorder='big', signed=False)
    uint32_full = uint32_msb + uint32_lsb
    uint32 = struct.unpack(">L",uint32_full)
    return uint32[0]

def uint16_to_int32(MSB,LSB):
    int32_msb = MSB.to_bytes(2, byteorder='big', signed=False)
    int32_lsb = LSB.to_bytes(2, byteorder='big', signed=False)
    int32_full = int32_msb + int32_lsb
    int32 = struct.unpack(">l",int32_full)
    return int32[0]
    
def uint16_to_float32(MSB,LSB):
    float32_msb = MSB.to_bytes(2, byteorder='big', signed=False)
    float32_lsb = LSB.to_bytes(2, byteorder='big', signed=False)
    float32_full = float32_msb + float32_lsb
    float32 = struct.unpack(">f",float32_full)
    return float32[0]

def uint16_to_uint64(MMSB,MLSB,LMSB,LLSB):
    uint64_mmsb = MMSB.to_bytes(2, byteorder='big', signed=False)
    uint64_mlsb = MLSB.to_bytes(2, byteorder='big', signed=False)
    uint64_lmsb = LMSB.to_bytes(2, byteorder='big', signed=False)
    uint64_llsb = LLSB.to_bytes(2, byteorder='big', signed=False)
    uint64_full = uint64_mmsb + uint64_mlsb + uint64_lmsb + uint64_llsb
    uint64 = struct.unpack(">Q",uint64_full)
    return uint64[0]

def uint32_to_uint64(MSB,LSB):
    uint64_msb = MSB.to_bytes(4, byteorder='big', signed=False)
    uint64_lsb = LSB.to_bytes(4, byteorder='big', signed=False)
    uint64_full = uint64_msb + uint64_lsb
    uint64 = struct.unpack(">Q",uint64_full)
    return uint64[0]

def uint16_to_hex_string(uintArray):
    hexStr=b''
    for uint in uintArray:
        hexStr+=uint.to_bytes(2, byteorder='big', signed=False)
    return hexStr.hex()
