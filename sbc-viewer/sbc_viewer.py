import sys
import os
import traceback
from pathlib import Path
import numpy as np
from typing import Dict, Any

# Set Qt plugin path before importing PySide6
try:
    import PySide6
    pyside6_dir = os.path.dirname(PySide6.__file__)
    plugins_path = os.path.join(pyside6_dir, 'plugins', 'platforms')
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugins_path
except ImportError:
    pass

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QTreeWidget, QTreeWidgetItem, QWidget, 
                               QFileDialog, QLabel, QHeaderView, QTextEdit, 
                               QTableWidget, QTableWidgetItem, QTabWidget, QMessageBox, 
                               QProgressBar, QStatusBar, QDialog)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor

# Import the SBC binary format reader
package_dir = Path(__file__).parent.parent
python_dir = package_dir / 'python'
if python_dir.exists():
    sys.path.insert(0, str(python_dir))

try:
    from sbcbinaryformat import Streamer
except ImportError:
    print("Warning: sbcbinaryformat module not found.")
    Streamer = None


class LoadFileWorker(QThread):
    """Simple worker thread for loading files."""
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
    
    def run(self):
        try:
            self.progress.emit("Loading file...")
            if Streamer is None:
                raise ImportError("SBC binary format reader not available")
            
            streamer = Streamer(self.file_path)
            data_dict = streamer.to_dict()
            self.finished.emit(data_dict)
        except Exception as e:
            self.error.emit(f"Error: {str(e)}")


class DataDialog(QDialog):
    """Simple dialog for viewing data details."""
    
    def __init__(self, name: str, data: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Data Viewer: {name}")
        self.resize(800, 600)
        
        layout = QVBoxLayout()
        
        # Info
        info = f"Variable: {name}\nType: {data.dtype}\nShape: {data.shape}\nSize: {data.size}"
        info_label = QLabel(info)
        info_label.setStyleSheet("background: #f8f9fa; padding: 10px; border: 1px solid #ddd;")
        layout.addWidget(info_label)
        
        # Create tabs for different views
        tabs = QTabWidget()
        
        # Table view for arrays
        if isinstance(data, np.ndarray) and data.size > 1:
            table = QTableWidget()
            table.setSortingEnabled(True)  # Enable header sorting
            self._populate_array_table(table, data)
            tabs.addTab(table, "Table View")
        
        # Statistics/data view
        if data.dtype.kind in ['i', 'u', 'f'] and data.size > 0:
            # Show statistics for numeric data
            text = QTextEdit()
            text.setReadOnly(True)
            text.setFont(QFont("Courier", 10))
            
            stats = []
            stats.append(f"Min: {np.min(data)}")
            stats.append(f"Max: {np.max(data)}")
            stats.append(f"Mean: {np.mean(data):.6f}")
            stats.append(f"Std: {np.std(data):.6f}")
            
            if data.size <= 1000:
                stats.append(f"\nData:\n{data}")
            else:
                stats.append(f"\nFirst 20 values:\n{data.flat[:20]}")
                stats.append(f"\nLast 20 values:\n{data.flat[-20:]}")
            
            text.setPlainText('\n'.join(stats))
            tabs.addTab(text, "Statistics")
        else:
            # Show raw data for other types
            text = QTextEdit()
            text.setReadOnly(True)
            text.setFont(QFont("Courier", 10))
            
            if data.size <= 100:
                text.setPlainText(str(data))
            else:
                text.setPlainText(f"Large array with {data.size} elements.\nFirst few: {data.flat[:10]}")
            
            tabs.addTab(text, "Raw Data")
        
        layout.addWidget(tabs)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        self.setStyleSheet("""
            QDialog { background: white; }
            QPushButton { background: #3498db; color: white; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background: #2980b9; }
            QTableWidget { 
                background: white; border: 1px solid #ddd; 
                gridline-color: #eee; 
            }
            QHeaderView::section { 
                background: #f8f9fa; border: 1px solid #ddd; 
                padding: 6px; font-weight: bold; 
            }
            QHeaderView::section:hover { background: #e9ecef; }
            QHeaderView::down-arrow, QHeaderView::up-arrow {
                width: 8px;
                height: 8px;
                margin-right: 3px;
                subcontrol-origin: padding;
                subcontrol-position: center right;
            }
        """)
    
    def _populate_array_table(self, table: QTableWidget, data: np.ndarray):
        """Populate table with all array data."""
        # Connect to sort signal to update row indices
        table.horizontalHeader().sortIndicatorChanged.connect(lambda: self._update_array_row_indices(table))
        
        if data.ndim == 1:
            # 1D array - single column, all rows
            table.setColumnCount(1)
            table.setRowCount(data.size)
            table.setHorizontalHeaderLabels(["Value"])
            
            for i in range(data.size):
                value = data[i]
                item = QTableWidgetItem(str(value))
                # Store numeric value for proper sorting and original index
                if data.dtype.kind in ['i', 'u', 'f']:
                    item.setData(Qt.UserRole, float(value))
                item.setData(Qt.UserRole + 1, i)  # Store original index
                table.setItem(i, 0, item)
                
        elif data.ndim == 2:
            # 2D array - all rows and columns (up to 500 columns for UI performance)
            rows, cols = data.shape
            max_display_cols = min(500, cols)  # Reasonable limit for UI
            
            table.setRowCount(rows)
            table.setColumnCount(max_display_cols)
            table.setHorizontalHeaderLabels([f"{i}" for i in range(max_display_cols)])
            
            for i in range(rows):
                for j in range(max_display_cols):
                    value = data[i, j]
                    item = QTableWidgetItem(str(value))
                    # Store numeric value for proper sorting and original index
                    if data.dtype.kind in ['i', 'u', 'f']:
                        item.setData(Qt.UserRole, float(value))
                    item.setData(Qt.UserRole + 1, i)  # Store original row index
                    table.setItem(i, j, item)
                    
        else:
            # Higher dimensional arrays - flatten and show all as single column
            flat_data = data.flatten()
            
            table.setColumnCount(1)
            table.setRowCount(len(flat_data))
            table.setHorizontalHeaderLabels(["Value (Flattened)"])
            
            for i in range(len(flat_data)):
                value = flat_data[i]
                item = QTableWidgetItem(str(value))
                # Store numeric value for proper sorting and original index
                if data.dtype.kind in ['i', 'u', 'f']:
                    item.setData(Qt.UserRole, float(value))
                item.setData(Qt.UserRole + 1, i)  # Store original index
                table.setItem(i, 0, item)
        
        # Set initial row labels
        self._update_array_row_indices(table)
        
        # Auto-resize columns to content (for reasonable sizes)
        if table.columnCount() <= 10 and table.rowCount() <= 1000:
            table.resizeColumnsToContents()
    
    def _update_array_row_indices(self, table: QTableWidget):
        """Update row indices to show original positions after sorting."""
        try:
            row_count = table.rowCount()
            if row_count == 0:
                return
            
            # Get original indices from the table items
            original_indices = []
            for row in range(row_count):
                item = table.item(row, 0)  # Get first column item
                if item:
                    original_index = item.data(Qt.UserRole + 1)
                    if original_index is not None:
                        original_indices.append(str(original_index))
                    else:
                        original_indices.append(str(row))
                else:
                    original_indices.append(str(row))
            
            # Update vertical header with original indices
            table.setVerticalHeaderLabels(original_indices)
        except Exception:
            # Fallback to sequential numbering if anything goes wrong
            row_count = table.rowCount()
            table.setVerticalHeaderLabels([str(i) for i in range(row_count)])


class NumericTableWidgetItem(QTableWidgetItem):
    """Custom table widget item that sorts numerically."""
    
    def __init__(self, text, numeric_value=None):
        super().__init__(text)
        self.numeric_value = numeric_value
    
    def __lt__(self, other):
        """Custom comparison for sorting."""
        if isinstance(other, NumericTableWidgetItem) and self.numeric_value is not None and other.numeric_value is not None:
            return self.numeric_value < other.numeric_value
        # Fall back to string comparison
        return super().__lt__(other)


class NumericTreeWidget(QTreeWidget):
    """Custom tree widget for consistent sorting."""
    
    def __init__(self, parent=None):
        super().__init__(parent)


class DataViewer(QWidget):
    """Main data viewer widget."""
    
    def __init__(self):
        super().__init__()
        self.data_dict = None
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Create tabs
        self.tabs = QTabWidget()
        
        # Tree view
        self.tree = NumericTreeWidget()
        self.tree.setHeaderLabels(["Variable", "Type", "Shape", "Preview"])
        self.tree.itemDoubleClicked.connect(self.on_tree_double_click)
        self.tabs.addTab(self.tree, "Tree View")
        
        # Table view  
        self.table = QTableWidget()
        self.table.itemDoubleClicked.connect(self.on_table_double_click)
        self.tabs.addTab(self.table, "Table View")
        
        # Text view
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        self.text.setFont(QFont("Courier", 10))
        self.tabs.addTab(self.text, "Text View")
        
        layout.addWidget(self.tabs)
        self.setLayout(layout)
    
    def set_data(self, data_dict: Dict[str, Any]):
        """Set data to display."""
        self.data_dict = data_dict
        self.populate_tree()
        self.populate_table()
        self.populate_text()
    
    def populate_tree(self):
        """Populate tree view."""
        self.tree.clear()
        self.tree.setSortingEnabled(True)  # Enable header sorting
        
        for name, data in self.data_dict.items():
            item = QTreeWidgetItem()
            item.setText(0, name)
            item.setText(1, str(data.dtype))
            item.setText(2, str(data.shape))
            item.setText(3, self._create_preview(data))
            item.setData(0, Qt.UserRole, data)
            
            self.tree.addTopLevelItem(item)
        
        # Auto-resize columns
        for i in range(4):
            self.tree.resizeColumnToContents(i)
    
    def _create_preview(self, data: np.ndarray) -> str:
        """Create a preview string showing the first few values."""
        if data.size == 0:
            return "Empty"
        elif data.size == 1:
            return str(data.item())
        elif data.size <= 5:
            if data.ndim == 1:
                return str(data.tolist())
            else:
                return str(data.flat[:data.size].tolist())
        else:
            if data.ndim == 1:
                # Show first few elements for 1D arrays
                first_elements = ', '.join(str(data[i]) for i in range(min(3, data.size)))
                return f"[{first_elements}, ...]"
            else:
                # For multi-dimensional arrays, show flattened first few
                flat_data = data.flat
                first_elements = ', '.join(str(flat_data[i]) for i in range(min(3, data.size)))
                return f"[{first_elements}, ...]"
    
    def populate_table(self):
        """Populate table view with all data."""
        if not self.data_dict:
            return
            
        # Load all rows
        max_rows = max(len(data) for data in self.data_dict.values()) if self.data_dict else 0
        
        if max_rows == 0:
            return
        
        self.table.setRowCount(max_rows)
        self.table.setColumnCount(len(self.data_dict))
        self.table.setHorizontalHeaderLabels(list(self.data_dict.keys()))
        self.table.setSortingEnabled(True)  # Enable header sorting
        
        # Connect to sort signal to update row indices
        self.table.horizontalHeader().sortIndicatorChanged.connect(self._update_row_indices)
        
        for col, (name, data) in enumerate(self.data_dict.items()):
            for row in range(min(max_rows, len(data))):
                value = data[row]
                
                if isinstance(value, np.ndarray):
                    if value.size <= 5:
                        text = str(value.tolist())
                    else:
                        text = f"Array{value.shape}"
                else:
                    text = str(value)
                
                # Use NumericTableWidgetItem for proper sorting of numeric values
                if isinstance(value, (int, float, np.number)) and not isinstance(value, np.ndarray):
                    try:
                        numeric_val = float(value)
                        item = NumericTableWidgetItem(text, numeric_val)
                    except:
                        item = QTableWidgetItem(text)
                else:
                    item = QTableWidgetItem(text)
                
                # Store original row index and other metadata
                item.setData(Qt.UserRole, {'value': value, 'name': name, 'row': row, 'original_row': row})
                
                self.table.setItem(row, col, item)
        
        # Set initial row labels
        self._update_row_indices()
    
    def _update_row_indices(self):
        """Update row indices to show original positions after sorting."""
        try:
            row_count = self.table.rowCount()
            if row_count == 0:
                return
            
            # Get original row indices from the table items
            original_indices = []
            for row in range(row_count):
                item = self.table.item(row, 0)  # Get first column item
                if item and item.data(Qt.UserRole):
                    data = item.data(Qt.UserRole)
                    if isinstance(data, dict) and 'original_row' in data:
                        original_indices.append(str(data['original_row']))
                    else:
                        original_indices.append(str(row))
                else:
                    original_indices.append(str(row))
            
            # Update vertical header with original row indices
            self.table.setVerticalHeaderLabels(original_indices)
        except Exception:
            # Fallback to sequential numbering if anything goes wrong
            row_count = self.table.rowCount()
            self.table.setVerticalHeaderLabels([str(i) for i in range(row_count)])
    
    def populate_text(self):
        """Populate text view."""
        lines = ["=== SBC Binary File Data ===\n"]
        
        for name, data in self.data_dict.items():
            lines.append(f"{name}:")
            lines.append(f"  Type: {data.dtype}")
            lines.append(f"  Shape: {data.shape}")
            lines.append(f"  Size: {data.size}")
            
            if data.size <= 10:
                lines.append(f"  Data: {data}")
            else:
                lines.append(f"  Preview: {data.flat[:5]}...{data.flat[-2:]}")
            lines.append("")
        
        self.text.setPlainText('\n'.join(lines))
    
    def on_tree_double_click(self, item, column):
        """Handle tree item double-click."""
        data = item.data(0, Qt.UserRole)
        if data is not None:
            dialog = DataDialog(item.text(0), data, self)
            dialog.show()
    
    def on_table_double_click(self, item):
        """Handle table item double-click."""
        data = item.data(Qt.UserRole)
        if data and isinstance(data['value'], np.ndarray):
            dialog = DataDialog(f"{data['name']}[{data['row']}]", data['value'], self)
            dialog.show()
    
    def clear_data(self):
        """Clear all data."""
        self.tree.clear()
        self.table.clear()
        self.text.clear()
        self.data_dict = None


class SBCViewer(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.load_worker = None
        self.setup_ui()
        self.setWindowTitle("SBC Binary Format Viewer")
        self.setGeometry(100, 100, 1000, 600)
        self.apply_styles()
    
    def setup_ui(self):
        """Setup the UI."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()
        
        # Header
        header = QHBoxLayout()
        
        self.load_btn = QPushButton("Load SBC File")
        self.load_btn.clicked.connect(self.load_file)
        self.load_btn.setMinimumHeight(35)
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: gray; font-style: italic;")
        
        header.addWidget(self.load_btn)
        header.addWidget(self.file_label)
        header.addStretch()
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        
        # Data viewer
        self.viewer = DataViewer()
        
        # Layout
        layout.addLayout(header)
        layout.addWidget(self.progress)
        layout.addWidget(self.viewer)
        central.setLayout(layout)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def apply_styles(self):
        """Apply simple, clean styling."""
        self.setStyleSheet("""
            QMainWindow, QWidget { background: white; color: #333; }
            QPushButton { 
                background: #3498db; color: white; border: none; 
                padding: 8px 16px; border-radius: 4px; font-weight: bold; 
            }
            QPushButton:hover { background: #2980b9; }
            QTableWidget, QTreeWidget { 
                background: white; border: 1px solid #ddd; 
                gridline-color: #eee; 
            }
            QHeaderView::section { 
                background: #f8f9fa; border: 1px solid #ddd; 
                padding: 6px; font-weight: bold; 
            }
            QHeaderView::down-arrow, QHeaderView::up-arrow {
                width: 8px;
                height: 8px;
                margin-right: 3px;
                subcontrol-origin: padding;
                subcontrol-position: center right;
            }
            QTextEdit { background: white; border: 1px solid #ddd; }
            QTabWidget::pane { background: white; border: 1px solid #ddd; }
            QTabBar::tab { 
                background: #f8f9fa; color: #333; padding: 8px 16px; 
                border: 1px solid #ddd; border-bottom: none; 
            }
            QTabBar::tab:selected { background: white; font-weight: bold; }
        """)
    
    def load_file(self):
        """Load a file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open SBC File", "", "SBC Files (*.bin *.sbc);;All Files (*)")
        
        if file_path:
            self.current_file = file_path
            self.file_label.setText(f"Loading: {Path(file_path).name}")
            self.load_btn.setEnabled(False)
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)  # Indeterminate
            self.viewer.clear_data()
            
            # Load in background
            self.load_worker = LoadFileWorker(file_path)
            self.load_worker.finished.connect(self.on_loaded)
            self.load_worker.error.connect(self.on_error)
            self.load_worker.progress.connect(self.statusBar().showMessage)
            self.load_worker.start()
    
    def on_loaded(self, data_dict):
        """Handle successful loading."""
        file_name = Path(self.current_file).name
        num_vars = len(data_dict)
        total_mb = sum(d.nbytes for d in data_dict.values()) / 1024 / 1024
        
        self.file_label.setText(f"Loaded: {file_name} ({num_vars} vars, {total_mb:.1f} MB)")
        self.load_btn.setEnabled(True)
        self.progress.setVisible(False)
        
        self.viewer.set_data(data_dict)
        self.statusBar().showMessage(f"Successfully loaded {num_vars} variables")
    
    def on_error(self, error_msg):
        """Handle loading error."""
        self.file_label.setText("Failed to load file")
        self.load_btn.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Error", error_msg)
        self.statusBar().showMessage("Error loading file")


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("SBC Binary Format Viewer")
    
    window = SBCViewer()
    window.show()
    
    if Streamer is None:
        QMessageBox.warning(window, "Missing Dependency", 
                          "SBC binary format library not found.\nEnsure 'python' directory is accessible.")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 