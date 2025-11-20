import numpy as np
import os

def sbcstring_to_type(type_str, endianess):
    out_type_str = ""
    if endianess == 'little':
        out_type_str += '<'
    elif endianess == 'big':
        out_type_str += '>'

    if type_str.startswith('string'):
        return np.dtype(out_type_str+type_str.replace("string", "U"))

    string_to_type = {'char': 'i1',
                      'int8': 'i1',
                      'int16': 'i2',
                      'int32': 'i4',
                      'int64': 'i8',
                      'uint8': 'u1',
                      'uint16': 'u2',
                      'uint32': 'u4',
                      'uint64': 'u8',
                      'single': 'f',
                      'float32': 'f',
                      'double': 'd',
                      'float64': 'd',
                      'float128': 'f16'}

    return np.dtype(out_type_str+string_to_type[type_str])


def type_to_sbcstring(sbc_type_str):
    if sbc_type_str.startswith("U"):
        return sbc_type_str.replace("U", "string")

    string_to_type = {'i1': 'int8',
                      'i2': 'int16',
                      'i4': 'int32',
                      'i8': 'int64',
                      'u1': 'uint8',
                      'u2': 'uint16',
                      'u4': 'uint32',
                      'u8': 'uint64',
                      'f': 'float32',
                      'd': 'double',
                      'f16': 'float128'}

    return string_to_type[sbc_type_str]

def is_gzip_file(filepath):
    """Check if a file is actually gzip compressed by reading magic bytes."""
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 2:
        return False
    with open(filepath, 'rb') as f:
        return f.read(2) == b'\x1f\x8b'
