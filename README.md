# SBCBinaryFormat

Here you will find all the code required to build and read a SBC binary format.

Languages supported:
* C++
* Python

## SBC Binary File Structure
The binary file is organized in the following sections. 

1. **Endianess (4 bytes, uint32)**
   - Used to indicate little or big endian.
   - Written as a hexadecimal constant (0x01020304 for little, 0x04030201 for big).

2. **Header Length (2 bytes, uint16)**
   - Size of the next segment (the Data Header) in bytes.

3. **Data Header (variable length)**
   - Contains semicolon-separated entries of the form:
     ```
     {column_name};{dtype};{dim1},{dim2}...
     ```
   - Dtype can be “int32”, “double”, etc.
   - Dimensions (e.g., “[3, 2]”) define the shape of each data element.

4. **Number of Lines (4 bytes, int32)**
   - Total number of records in the file when known.
   - Set to zero for an open-ended file.

5. **Data Lines**
   - Each line is a concatenation of binary values for each column.
   - Each column is written in the correct dtype and shape specified in the header.