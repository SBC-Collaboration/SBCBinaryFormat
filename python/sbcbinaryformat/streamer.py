import os
import numpy as np
from collections import OrderedDict
from .utilities import sbcstring_to_type, type_to_sbcstring
import warnings

class Streamer:
    """
    This class manages opening a sbc binary file. It reads the header and 
    saves data into a dictionary of numpy arrays. For large files, it uses
    block-based loading to handle partial reads efficiently.
    
    :param file_name: Path to the SBC binary file
    :type file_name: str
    :param max_size: Maximum size in MB to load at once (0 = load all)
    :type max_size: int
    :param block_len: Number of rows per block for streaming mode
    :type block_len: int
    :raises OSError: If the file format is invalid or unsupported endianness
    :raises ValueError: If the header format is incorrect or incompatible
    :raises IndexError: If the requested range is out of bounds
    :raises NotImplementedError: If step slicing is attempted in streaming mode
    :raises ValueError: If the 'end' and 'length' parameters are used together
    :raises ValueError: If the 'start' parameter is negative and exceeds file length
    :raises ValueError: If the 'length' parameter is not positive
    :raises ValueError: If the 'end' parameter is negative and exceeds file length
    """
    
    def __init__(self, file_name, max_size=0, block_len=100):
        self.file_name = file_name
        self.file_size = os.path.getsize(file_name)
        self.max_size_bytes = max_size * 1024 * 1024
        self.block_len = block_len

        # Parse header and setup dtype
        self._parse_header()
        
        # Determine if we can load everything at once
        self.single_read = (self.file_size < self.max_size_bytes or max_size == 0)
        
        if self.single_read:
            # Load all data immediately
            self.data = np.fromfile(self.file_name, dtype=self.row_dtype, offset=self.header_size)
            self.file_handle = None
        else:
            # Setup for block-based reading
            self.file_handle = open(self.file_name, "rb")
            self.data = None
    
    def _parse_header(self):
        """Parse the SBC binary header to extract column information"""
        with open(self.file_name, "rb") as f:
            # Read endianness
            endian_val = np.fromfile(f, dtype=np.uint32, count=1)[0]
            if endian_val == 0x01020304:
                self.endianness = "little"
            elif endian_val == 0x04030201:
                self.endianness = "big"
            else:
                raise OSError(f"Unsupported endianness: {endian_val}")
            
            # Read header
            header_length = np.fromfile(f, dtype=np.uint16, count=1)[0]
            header = f.read(header_length).decode('ascii')
            header_items = header.split(';')[:-1]  # Remove empty last element
            
            if len(header_items) % 3 != 0:
                raise OSError("Invalid header format")
            
            # Parse columns
            num_columns = len(header_items) // 3
            self.columns = []
            fields = []
            
            for i in range(num_columns):
                name = header_items[i * 3]
                type_str = header_items[i * 3 + 1]
                size_str = header_items[i * 3 + 2]
                
                self.columns.append(name)
                dtype = self._parse_dtype(type_str)
                sizes = list(map(int, size_str.split(',')))
                
                # Build numpy dtype field
                if sizes == [1]:
                    fields.append((name, dtype))
                else:
                    fields.append((name, dtype, tuple(sizes)))
            
            self.row_dtype = np.dtype(fields)
            
            # Skip expected number of elements (we'll calculate it)
            f.seek(4, 1)
            self.header_size = f.tell()
            
            # Calculate actual number of elements
            payload_size = self.file_size - self.header_size
            if payload_size % self.row_dtype.itemsize != 0:
                # Instead of raising an error, warn the user and proceed.
                warnings.warn(
                    f"File '{self.file_name}' may be truncated or corrupted. "
                    "The last partial row will be ignored.",
                    UserWarning
                )
            
            self.num_elems = payload_size // self.row_dtype.itemsize
            self.block_len = min(self.block_len, self.max_size_bytes // self.row_dtype.itemsize)
    
    def _parse_dtype(self, type_str):
        """Convert SBC type string to numpy dtype"""
        if type_str.startswith('string'):
            endian_char = '<' if self.endianness == 'little' else '>'
            return np.dtype(endian_char + type_str.replace("string", "U"))
        
        return sbcstring_to_type(type_str, self.endianness) 
    
    def __getitem__(self, idx):
        """Get data by index or slice"""
        if self.single_read:
            return self.data[idx]
        else:
            # Handle streaming mode
            if isinstance(idx, (int, np.integer)):
                if idx < 0:
                    idx = self.num_elems + idx
                return self._get_block_data(idx, idx + 1)[0]
            
            elif isinstance(idx, slice):
                start, stop, step = idx.indices(self.num_elems)
                if step != 1:
                    raise NotImplementedError("Step slicing not supported in streaming mode")
                return self._get_range_data(start, stop)
            
            else:
                raise TypeError(f"Unsupported index type: {type(idx)}")
    
    def _get_range_data(self, start, stop):
        """Get data for a range, potentially spanning multiple blocks"""
        if stop - start <= self.block_len:
            # Single block read
            return self._get_block_data(start, stop)
        
        # Multi-block read - concatenate blocks
        data_parts = []
        current = start
        
        while current < stop:
            block_end = min(current + self.block_len, stop)
            block_data = self._get_block_data(current, block_end)
            data_parts.append(block_data)
            current = block_end
        
        return np.concatenate(data_parts)
    
    def _get_block_data(self, start, stop):
        """Get data for a single block"""
        # Load block from file
        block_start = start
        block_end = min(block_start + self.block_len, self.num_elems)
        
        byte_offset = self.header_size + block_start * self.row_dtype.itemsize
        self.file_handle.seek(byte_offset)
        
        block_data = np.fromfile(
            self.file_handle, 
            dtype=self.row_dtype, 
            count=block_end - block_start
        )
        
        # Return requested slice
        local_start = start - block_start
        local_stop = local_start + (stop - start)
        return block_data[local_start:local_stop]
    
    def to_dict(self, start=None, end=None, length=None):
        """
        Convert data to dictionary of arrays
        
        :param start: Start index for range (inclusive). If not provided, defaults to the beginning of the data.
        :param end: End index for range (exclusive). If not provided, defaults to the end of the data.
        :param length: Number of elements to read starting from start. Cannot be used together with end.
        """
        # Validate parameter combinations
        if end is not None and length is not None:
            raise ValueError("Cannot specify both 'end' and 'length' parameters")
        
        # Handle start/end/length parameters
        if start is not None or end is not None or length is not None:
            if start is None:
                start = 0
            
            # Calculate end based on length if provided
            if length is not None:
                if length <= 0:
                    raise ValueError("Length must be positive")
                end = start + length
            elif end is None:
                end = self.num_elems
            
            # Validate indices
            if start < 0:
                start = self.num_elems + start
            if end < 0:
                end = self.num_elems + end
            if start < 0 or end > self.num_elems or start >= end:
                raise IndexError(f"Invalid range: start={start}, end={end} for length {self.num_elems}")
            data = self[start:end]
        else:
            data = self[:]

        return OrderedDict({name: np.array(data[name]) for name in self.columns})

    def __len__(self):
        return self.num_elems
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.file_handle:
            self.file_handle.close()
    
    # Utility methods for inspection
    def get_info(self):
        """Get file and streaming information"""
        return {
            'file_size': self.file_size,
            'num_elements': self.num_elems,
            'columns': self.columns,
            'single_read': self.single_read,
            'block_size': self.block_len if not self.single_read else self.num_elems
        }
