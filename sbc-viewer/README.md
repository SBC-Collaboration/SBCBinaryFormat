# SBC Binary Format Viewer

A modern PySide6-based GUI application for viewing and exploring SBC binary format files, similar to Spyder's variable explorer.

## Features

- **Modern GUI Interface**: Clean, intuitive interface with multiple viewing modes
- **Multiple View Modes**: 
  - Tree View: Hierarchical display of variables with type information and color-coding
  - Table View: Spreadsheet-like view for easy data inspection  
  - Text View: Detailed text representation with statistics
- **File Loading**: Simple file browser to load `.sbc.bin` files
- **Data Type Support**: Full support for all SBC binary format data types (integers, floats, strings, arrays)
- **Performance**: Efficient loading with background processing and progress indication
- **Data Exploration**: 
  - Double-click any variable for detailed view
  - Color-coded data types (blue for integers, green for floats, orange for strings)
  - Statistical information for numeric data
  - Array preview and truncation for large datasets
- **Spyder-style Color Coding**: Heatmap visualization based on data values

## Installation

### Prerequisites

1. Python 3.8 or higher
2. Required Python packages (install via pip)

### Setup

1. **Clone or download this repository** to your local machine

2. **Install the package**:
   ```bash
   cd sbc-viewer
   pip install -e .
   ```
   
   Or install from the repository root:
   ```bash
   pip install -e ./sbc-viewer
   ```

3. **Ensure SBC library is available**:
   The application expects the `python/sbcbinaryformat.py` file from the parent repository to be accessible.

## Usage

### Running the Application

1. **Command Line (after installation)**:
   ```bash
   sbc-viewer
   # or
   sbc-viewer-gui
   ```

2. **Development mode**:
   ```bash
   cd sbc-viewer
   python -m sbc_viewer.main
   ```

3. **The main window will open** with:
   - "Load SBC File" button in the top-left
   - File information display
   - Tabbed data viewer (Tree View, Table View, Text View)

### Loading Files

1. Click the **"Load SBC File"** button
2. Browse and select your `.sbc.bin` file
3. The application will load the file in the background
4. Progress will be shown in the status bar
5. Once loaded, data will appear in all three view tabs

### Viewing Data

#### Tree View
- Shows all variables in a hierarchical list
- Columns: Variable Name, Data Type, Shape, Size, Value Preview
- Color-coded by data type:
  - **Blue background**: Integer types (int8, int16, int32, int64, uint8, etc.)
  - **Green background**: Float types (float32, float64, double)
  - **Orange background**: String types
  - **Gray background**: Other types
- **Double-click** any row to see detailed information

#### Table View
- Spreadsheet-style view of the data
- Each column represents a variable
- Each row represents a record/observation
- **Spyder-style color coding** with toggle button:
  - Blue to red heatmap for numeric data (column-relative)
  - Green gradient for string data (by length)
- Limited to first 1000 rows for performance
- Arrays and complex data shown as summaries
- **Double-click** any cell for detailed inspection

#### Text View
- Detailed text representation of all variables
- Includes statistical information for numeric data (min, max, mean, std)
- Shows data previews and array information
- Copy-friendly format for documentation

### Data Types Supported

The viewer supports all SBC binary format data types:
- **Integers**: int8, int16, int32, int64, uint8, uint16, uint32, uint64
- **Floats**: float32, float64, double, float128
- **Strings**: string types of various lengths
- **Arrays**: Multi-dimensional arrays of any supported type
- **Mixed Data**: Files with different data types per column

## File Format Support

This viewer is specifically designed for **SBC Binary Format** files as defined in the parent repository. The format includes:

1. **Endianness indicator** (4 bytes)
2. **Header length** (2 bytes) 
3. **Data header** (variable length) - semicolon-separated column definitions
4. **Number of records** (4 bytes)
5. **Binary data** - structured according to header definitions

## Example Files

The parent repository includes `python/example.sbc.bin` which you can use to test the viewer and see the different data types in action.

## Development

### Installing for Development

```bash
git clone <repository-url>
cd sbc-viewer
pip install -e .[dev]
```

### Code Structure

- **`sbc_viewer/main.py`**: Main application file
- **`LoadFileWorker`**: Background thread for file loading
- **`DataViewer`**: Main data display widget with tabs
- **`SBCViewer`**: Main application window
- **`DetailedDataDialog`**: Popup dialog for detailed array inspection
- **`ScalarDetailedDialog`**: Popup dialog for scalar value inspection

### Dependencies

- **PySide6**: Qt-based GUI framework
- **NumPy**: Numerical computing library  
- **sbcbinaryformat**: SBC binary format reader (from parent repository)

### Development Tools

The package includes development dependencies:
- **pytest**: For testing
- **pytest-qt**: For GUI testing
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking

Run tests:
```bash
pytest
```

Format code:
```bash
black sbc_viewer/
```

## Troubleshooting

### Common Issues

1. **"SBC binary format library not available" warning**:
   - Ensure the parent `python/` directory with `sbcbinaryformat.py` is accessible
   - Check the relative path from the package installation

2. **File loading errors**:
   - Verify the file is a valid SBC binary format file
   - Check file permissions
   - Ensure the file isn't corrupted

3. **Performance issues with large files**:
   - The table view limits display to 1000 rows for performance
   - Very large files (>1GB) may take time to load
   - Use the Tree or Text view for better performance with large datasets

### Error Messages

The application provides detailed error messages in popup dialogs. Common errors include:
- Invalid file format
- Corrupted headers
- Unsupported data types
- File access permissions

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the viewer functionality.

## License

This project follows the same license as the parent SBC Binary Format repository. 