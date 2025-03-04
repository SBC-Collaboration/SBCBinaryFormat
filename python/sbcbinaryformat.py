"""
Contains two classes:

* Streamer: which reads a file and either creates a streamer
  or saved everything to RAM. No multiprocessing features.
* Writer: which creates SBC binary files given the passed parameters.
Very simplistic and does not have any multiprocessing features.
"""
import sys
import os
import numpy as np


class Streamer:
    """
    This class manages opening a sbc binary file. It reads the header and 
    saves data into a dictionary of numpy arrays.
    """
    def __init__(self, file, max_size=1000000000):
        self.system_endianess = sys.byteorder
        self.file_size = os.path.getsize(file)
        self.file = file
        self.is_all_in_ram = self.file_size < max_size

        # Read header info
        with open(file, "rb") as f:
            # Read and check endianess
            file_endianess_val = np.fromfile(f, dtype=np.uint32, count=1)[0]
            if file_endianess_val == 0x01020304:
                self.file_endianess = "little"
            elif file_endianess_val == 0x04030201:
                self.file_endianess = "big"
            else:
                raise OSError(f"Endianess not supported: {file_endianess_val}")

            # Read header length and header string
            self.header_length = int(np.fromfile(f, dtype=np.uint16, count=1)[0])
            header = f.read(self.header_length).decode('ascii')
            header_items = header.split(';')[:-1]  # last element is empty

            if len(header_items) % 3 != 0:
                raise OSError("Header format error: items not in multiples of 3")

            num_columns = len(header_items) // 3
            self.columns = []
            self.dtypes = []
            self.sizes = []
            for i in range(num_columns):
                name = header_items[i * 3]
                type_str = header_items[i * 3 + 1]
                size_str = header_items[i * 3 + 2]
                self.columns.append(name)
                self.dtypes.append(sbcstring_to_type(type_str, self.file_endianess))
                self.sizes.append(list(map(int, size_str.split(','))))

            # Read the expected number of elements (can be 0 if unknown)
            self.expected_num_elems = np.fromfile(f, dtype=np.int32, count=1)[0]
            self.header_size = f.tell()

        # Create a structured dtype from the header info
        fields = []
        for col, dtype, sizes in zip(self.columns, self.dtypes, self.sizes):
            # If sizes is [1], store as scalar; otherwise as a subarray.
            if sizes == [1]:
                fields.append((col, dtype))
            else:
                fields.append((col, dtype, tuple(sizes)))
        self.row_dtype = np.dtype(fields)

        # Calculate the number of rows by comparing the payload size to the row size.
        self.payload_size = self.file_size - self.header_size
        if self.payload_size % self.row_dtype.itemsize != 0:
            raise OSError("File payload size does not match structured row size")
        self.num_elems = self.payload_size // self.row_dtype.itemsize
        if self.expected_num_elems and self.expected_num_elems != self.num_elems:
            raise OSError("Expected number of elements does not match calculated number")

        # Read the data.
        self.data = np.fromfile(file, dtype=self.row_dtype, offset=self.header_size)

    def __getitem__(self, idx):
        return self.data[idx]

    def to_dict(self):
        # Convert the structured array into a dictionary for easier access.
        return {name: self.data[name] for name in self.columns}


class Writer:
    """
    SBC Binary Header description:
     * Header of a binary format is divided in 4 parts:
     * 1.- Edianess            - always 4 bits long (uint32_t)
     * 2.- Data Header size    - always 2 bits long (uint16_t)
     * and is the length of the next bit of data
     * 3.- Data Header         - is data header long.
     * Contains the structure of each line. It is always found as a raw
     * string in the form "{name_col};{type_col};{size1},{size2}...;...;
     * Cannot be longer than 65536 bytes.
     * 4.- Number of lines     - always 4 bits long (int32_t)
     * Number of lines in the file. If 0, it is indefinitely long.
    """

    def __init__(self, file_name, columns_names, dtypes, sizes):
        self.file_name = file_name
        self.num_elems_saved = 0
        self.system_endianess = sys.byteorder

        if len(columns_names) != len(dtypes):
            raise ValueError("columns names and dtypes should be of the \
same length")

        # if does not exist OR its size is 0, we create the header
        if not os.path.exists(file_name) or os.path.getsize(file_name) == 0:
            self.__create_header(file_name, columns_names, dtypes, sizes)
        # Otherwise, we read the header and check whenever is compatible with
        # current parameters
        else:
            if not self.__has_compatible_header(file_name, columns_names,
                                                dtypes, sizes):
                raise ValueError(f"Header of already existing file must match \
columns_names ({columns_names}), dtypes ({dtypes}), \
and sizes ({sizes})")

            self.file_resource = open(file_name, 'ab')

    def __len__(self):
        return self.num_elems_saved

    def __enter__(self):
        """
            This allows the use of with()
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
            This allows the use of with() and properly closes the resources
        """
        self.file_resource.close()

    def write(self, data):
        """
        Write multi line data to file.

        :raises Value Error: if data is not a dictioanry or a list of dictionaries
        or keys or lengths do not match columns of the file,
        or dimensions of values cannot be broadcasted to sizes.
        """
        if isinstance(data, list):
            for d in data:
                self.write(d)
            return

        # At this point, data is assumed to be a dictionary.
        keylist = list(self.columns)
        if list(data.keys()) != keylist:
            raise ValueError(f"Data keys must match the file's column names: {keylist}, "
                            f"but got {list(data.keys())}")

        # Convert each column’s data to a NumPy array.
        arrays = {}
        nrows_list = []
        for k in keylist:
            # data_column
            dcol = np.asarray(data[k])
            # remove all dimensions of size 1
            dcol = np.squeeze(dcol)
            # If dcol is a scalar, make it a one-element array.
            if dcol.ndim == 0:
                dcol = np.array([dcol])
            # For columns expected to be scalars (sizes == [1]), squeeze extra dimensions.
            idx = keylist.index(k)
            exp_size = self.sizes[idx]
            real_size = list(dcol.shape)
            
            # check if the sizes are compatible
            # if exp_size is [a, b, ...]
            # a must have shape [a, b, ...] or [nrows, a, b, ...]
            # if exp_size is [1]
            # dcol must have shape [1] or [nrows]
            if real_size == exp_size:
                # if there is only one row, and the size is the same
                nrows_list.append(1)
            elif real_size[1:] == exp_size:
                # if after removing first dim, the sizes is same
                nrows_list.append(real_size[0])
            elif exp_size == [1] and len(real_size) == 1:
                # if the size is [1] and the shape is [nrows]
                nrows_list.append(real_size[0])
            else: raise ValueError(f"Data for column '{k}' has unexpected shape {real_size}, expected {exp_size}.")

        # check if number of rows for each column is consistent
        # they should all be the same or is 1
        nrows = max(nrows_list)
        if not all(nrows == x or x == 1 for x in nrows_list):
            sizes_dict = {str(k): v for k, v in zip(keylist, nrows_list)}
            raise ValueError(f"All columns must have the same number of rows. Got sizes {sizes_dict}")
        else:
            for idx in range(len(nrows_list)):
                k = keylist[idx]
                exp_size = self.sizes[idx]
                if nrows_list[idx] == 1:
                    arrays[k] = np.broadcast_to(data[k], (nrows,) +tuple(exp_size))
                else:
                    arrays[k] = data[k]

        # Build a structured dtype for one row.
        # For each column, if the expected size is [1] we treat it as a scalar;
        # otherwise we use a subarray field.
        fields = []
        for k, col_dtype, col_sizes in zip(keylist, self.dtypes, self.sizes):
            np_dtype = np.dtype(col_dtype)
            if col_sizes == [1]:
                fields.append((k, np_dtype))
            else:
                fields.append((k, np_dtype, tuple(col_sizes)))
        # Ensure there is no padding between fields.
        rec_dtype = np.dtype(fields, align=False)

        # Create a structured array for all rows.
        rec = np.empty(nrows, dtype=rec_dtype)
        for k in keylist:
            idx = keylist.index(k)
            expected_sizes = self.sizes[idx]
            if expected_sizes == [1]:
                expected_shape = (nrows,)
            else:
                expected_shape = (nrows,) + tuple(expected_sizes)
            if arrays[k].shape != expected_shape:
                try:
                    arrays[k] = arrays[k].reshape(expected_shape)
                except Exception as e:
                    raise ValueError(f"Data for column '{k}' cannot be reshaped to {expected_shape}") from e
            rec[k] = arrays[k]

        # Write the entire block at once.
        rec.tofile(self.file_resource)
        self.num_elems_saved += nrows

    def __create_header(self, file_name, columns_names, dtypes, sizes):
        self.file_resource = open(file_name, 'wb')
        # first endianess
        np.array(0x01020304, dtype='u4').tofile(self.file_resource)

        # then we need two things: the header and its length
        self.header = ""
        for column_name, dtype, size in zip(columns_names, dtypes, sizes):
            header_str = ""
            if isinstance(size, int):
                header_str = f"{size}"
            else:
                for i, size_i in enumerate(size):
                    if i == 0:
                        header_str += f"{size_i}"
                    else:
                        header_str += f",{size_i}"

            self.header += f"{column_name};{type_to_sbcstring(dtype)};{header_str};"

        self.header_length = len(self.header)
        np.array(self.header_length, dtype='u2').tofile(self.file_resource)

        self.file_resource.write(self.header.encode('ascii'))

        np.array(0, dtype='i4').tofile(self.file_resource)

        self.columns = np.array(columns_names)
        self.dtypes = np.array(dtypes)
        self.sizes = sizes

    def __has_compatible_header(self, file_name, columns_names, dtypes, sizes):
        # open the file for read only
        with open(file_name, "rb") as file:
            file_endianess = np.fromfile(file,
                                         dtype=np.uint32, count=1)[0]

            if file_endianess not in (0x1020304, 0x4030201):
                raise OSError(f"Endianess not supported: 0x{file_endianess:X}")

            self.correct_for_endian = False
            if (file_endianess == 0x04030201
                and self.system_endianess == 'little') \
               or (file_endianess == 0x01020304
                   and self.system_endianess == 'big'):
                self.correct_for_endian = True

            # Now the length of the header
            self.header_length = np.fromfile(file,
                                             dtype=np.uint16,
                                             count=1)[0]

            header = file.read(self.header_length).decode('ascii')
            header = header.split(';')

            if (len(header) - 1) % 3 != 0:
                raise OSError(f"The number of items found in the header \
                    should always come in multiples of 3. It is \
                    {len(header) - 1}")

            self.num_columns = int(len(header) / 3)
            header = np.resize(header, (self.num_columns, 3))

            self.columns = header[:, 0]

            if not (self.columns == columns_names).all():
                print(f"SBCWriter: columns did not match {self.columns} and \
                    {columns_names}")
                return False

            self.dtypes = np.array([sbcstring_to_type(type_s, file)
                           for type_s in header[:, 1]])

            np_dtypes = [np.dtype(dtype) for dtype in dtypes]
            if not (self.dtypes == np_dtypes).all():
                print("SBCWriter: dtypes did not match")
                return False

            self.sizes = [[int(lenght) for lenght in length.split(',')]
                          for length in header[:, 2]]

            if not self.__check_sizes(sizes):
                print("SBCWriter: sizes did not match")
                return False

            self.num_elems = np.fromfile(file,
                                         dtype=np.int32,
                                         count=1)[0]
        # If it passed all the test, then they match!
        return True

    def __check_sizes(self, sizes):
        return np.array([(np.array(x) == y).all() \
                         for x, y in zip(self.sizes, sizes)]).all()


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

if __name__ == "__main__":
    # Writer example
    with Writer(
        "example.sbc.bin",
        ["t", "x", "y", "z", "momentum", "source"],
        ['i4', 'd', 'd', 'd', 'd', "U100"],
        [[1], [1], [1], [1], [3, 2], [1]]) as example_writer:

        example_writer.write({
            't': [1],
            'x': [2.0],
            'y': [3.0],
            'z': [4.0],
            'momentum': [[1, 2], [4, 5], [7, 8]],
            'source': ["Bg"]})

        n_lines = 10
        rng = np.random.default_rng()

        data = {
            't': rng.integers(-10, 10, (n_lines)),
            'x': 2,
            'y': rng.random((n_lines)),
            'z': rng.random((n_lines)),
            'momentum': rng.random((n_lines, 3, 2)),
            'source': rng.choice(["Bg", "Co-60", "Th-228", "Sb-124"], (n_lines))}
        
        example_writer.write(data)
    
    # Reader example
    example_streamer = Streamer("example.sbc.bin")
    data_dict = example_streamer.to_dict()
    for key, value in data_dict.items():
        print(f"key: {key}\t shape: {value.shape}")
        print(value[-10:])
