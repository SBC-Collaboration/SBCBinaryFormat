# Python SBC Binary Format driver

Use this library either directly (copy sbcbinaryformat.py into your source code) or install it as a library. 

## How to install using `pip`
1. Clone this repository to your local machine: `git clone https://github.com/SBC-Collaboration/SBCBinaryFormat.git`.
2. (Optional) If using conda environments, activate your target environment.
3. Make sure dependencies such as `pip`, `wheel`, `setuptools`, and `numpy` are installed.
4. Within the `python` folder, run the following command
```
pip install .
```

## Copy without installing
1. Copy `sbcbinaryformat.py` into the same directory as your code file.
2. OR: Add the path to `sbcbinaryformat.py` to your `PYTHONPATH` by running `export PYTHONPATH=$PYTHONPATH:/path/to/the/file/`.

## How to use
Here's example usage of both the `Writer` and `Streamer`. They are also at the end of `sbcbinaryformat`, which can be run by running the library on its own.

### Writer example
The `Writer` class initializes with 4 parameters:
1. File name (string).
2. Column names (list of strings).
3. Data types (list of dtypes). The supported data types are:
    - Signed integers (`i1`: int8, `i2`: int16, `i4`: int32, `i8`: int64)
    - Unsigned integers (`u1`: uint8, `u2`: uint16, `u4`: uint32, `u8`: uint64)
    - Floats (`f`: float32, `d`: double/float64: `f16`: float128)
    - Strings (`U{len}`: unicode string of maximum length `len`)
4. Dimensions (list of lists of integers).
```python
from sbcbinaryformat import Writer

file_name = "example.sbc.bin"
column_names = ["t", "x", "y", "z", "momentum", "source"]
data_types = ['i4', 'd', 'd', 'd', 'd', "U100"]
sizes = [[1], [1], [1], [1], [3, 2], [1]]

example_writer = Writer(file_name, column_names, data_types, sizes)
```
Then prepare data into a dictionary, matching the description provided above. Use `Writer.write()` to write the data to file. If the files doesn't exist, it will be created. Writing multiple times will append to the end of file.
```python
data = {
    't': [1],
    'x': [2.0],
    'y': [3.0],
    'z': [4.0],
    'momentum': [[1, 2], [4, 5], [7, 8]],
    'source': ["Bg"]}
example_writer.write(data)
```
Each `Writer.write()` call can write a dictionary with multiple lines, or a list of dictionaries. The first dimension (the number of rows) of each value in the dictionary needs to be the same across all columns. The rest of the dimension needs to match the dimension defined by `sizes` in the initialization. If any dimension has shape 1, it will be squeezed. If the shape of one column is 1, it can be broadcast to fit the shape of other columns.
```python
data = {
    't': [1, 2, 3],
    'x': 2.0,
    'y': [3.0, 2.0, 1.0],
    'z': [4.0, 4.0, 4.0],
    'momentum': [
        [[1, 2], [4, 5], [7, 8]],
        [[1, 1], [2, 2], [3, 3]],
        [[3, 3], [2, 2], [1, 1]]],
    'source': ["Bg", "Th-228", "Th-228"]}
example_writer.write(data)
```

### Reader example
The `Streamer` class can be initialized with only the path to the binary file. Use `Streamer.to_dict()` to convert the data into a python dictionary for easier handling.
```python
from sbcbinaryformat import Streamer

example_streamer = Streamer("example.sbc.bin")
data_dict = example_streamer.to_dict()
for key, value in data_dict.items():
    print(f"key: {key}\t shape: {value.shape}")
    print(value[:10])
```