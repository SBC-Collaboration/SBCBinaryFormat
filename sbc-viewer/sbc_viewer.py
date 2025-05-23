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
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QObject
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


class BaseWorker(QThread):
    """Base worker thread with common signals and interruption."""
    batch_ready = Signal(int, list)
    finished = Signal()
    progress = Signal(str, int)
    error = Signal(str)
    _worker_id_counter = 0

    def __init__(self):
        super().__init__()
        self._is_interruption_requested = False
        BaseWorker._worker_id_counter += 1
        self.worker_id = BaseWorker._worker_id_counter

    def requestInterruption(self):
        self._is_interruption_requested = True

    def run(self):
        try:
            total_r = self._get_total_rows_for_run()
            if total_r == 0:
                self.finished.emit()
                return

            current_row = self._get_start_row_for_run()
            batch_size = self._get_batch_size_for_run()
            end_row = min(current_row + batch_size, total_r)

            if self._is_interruption_requested: 
                return
            
            self.progress.emit(f"Loading data from row {current_row} to {end_row}...", 0)
            
            batch_data = self._process_batch(current_row, end_row)

            if batch_data is None:
                return

            if self._is_interruption_requested: 
                return
            self.batch_ready.emit(current_row, batch_data)

            if self._is_interruption_requested: 
                return
            self.progress.emit("Batch complete!", 100)

            if self._is_interruption_requested: 
                return
            self.finished.emit()

        except Exception as e:
            traceback.print_exc()
            if not self._is_interruption_requested:
                self.error.emit(f"Error loading data: {str(e)}")

    def _get_total_rows_for_run(self):
        raise NotImplementedError
    
    def _get_start_row_for_run(self):
        raise NotImplementedError

    def _get_batch_size_for_run(self):
        raise NotImplementedError

    def _process_batch(self, current_row: int, end_row: int) -> list | None:
        raise NotImplementedError


class ArrayLoadWorker(BaseWorker):
    """Worker thread for loading array data in background."""
    
    def __init__(self, data: np.ndarray, start_row: int = 0, batch_size: int = 2000):
        super().__init__() 
        self.data = data
        self.start_row = start_row
        self.batch_size = batch_size
        self.total_rows = self._get_total_rows_from_data()
    
    def _get_total_rows_from_data(self):
        """Get total number of rows to display."""
        if self.data.ndim == 1:
            return self.data.size
        elif self.data.ndim == 2:
            return self.data.shape[0]
        else:
            return self.data.size  # Flattened

    # Implementation of abstract methods from BaseWorker
    def _get_total_rows_for_run(self):
        return self.total_rows

    def _get_start_row_for_run(self):
        return self.start_row

    def _get_batch_size_for_run(self):
        return self.batch_size

    def _process_batch(self, current_row: int, end_row: int) -> list | None:
        batch_data = []
        for row in range(current_row, end_row):
            if self._is_interruption_requested:
                return None
            
            row_data = []
            
            if self.data.ndim == 1:
                if row < self.data.size:
                    value = self.data[row]
                    row_data.append({
                        'text': str(value),
                        'value': value,
                        'original_index': row,
                        'is_numeric': self.data.dtype.kind in ['i', 'u', 'f']
                    })
            elif self.data.ndim == 2:
                max_cols = min(500, self.data.shape[1])
                if row < self.data.shape[0]:
                    for col in range(max_cols):
                        value = self.data[row, col]
                        row_data.append({
                            'text': str(value),
                            'value': value,
                            'original_index': row,
                            'is_numeric': self.data.dtype.kind in ['i', 'u', 'f']
                        })
                else:
                    for col in range(max_cols):
                        row_data.append({
                            'text': '',
                            'value': None,
                            'original_index': row,
                            'is_numeric': False
                        })
            else:
                flat_data = self.data.flatten()
                if row < len(flat_data):
                    value = flat_data[row]
                    row_data.append({
                        'text': str(value),
                        'value': value,
                        'original_index': row,
                        'is_numeric': self.data.dtype.kind in ['i', 'u', 'f']
                    })
            
            batch_data.append(row_data)
        return batch_data


class TableLoadWorker(BaseWorker):
    """Worker thread for loading table data in background with lazy loading."""
    
    def __init__(self, data_dict: Dict[str, Any], start_row: int = 0, batch_size: int = 1000):
        super().__init__()
        self.data_dict = data_dict
        self.start_row = start_row
        self.batch_size = batch_size
        self.total_rows = max(len(data) for data in data_dict.values()) if data_dict else 0

    # Implementation of abstract methods from BaseWorker
    def _get_total_rows_for_run(self):
        return self.total_rows

    def _get_start_row_for_run(self):
        return self.start_row

    def _get_batch_size_for_run(self):
        return self.batch_size

    def _process_batch(self, current_row: int, end_row: int) -> list | None:
        batch_data = []
        if not self.data_dict:
            return batch_data

        for row in range(current_row, end_row):
            if self._is_interruption_requested:
                return None

            row_data = []
            for col, (name, data) in enumerate(self.data_dict.items()):
                if row < len(data):
                    value = data[row]
                    
                    if isinstance(value, np.ndarray):
                        if value.size <= 5:
                            text = str(value.tolist())
                        else:
                            text = f"Array{value.shape}"
                    else:
                        text = str(value)
                    
                    cell_data = {
                        'text': text,
                        'value': value,
                        'name': name,
                        'row': row,
                        'original_row': row,
                        'is_numeric': isinstance(value, (int, float, np.number)) and not isinstance(value, np.ndarray)
                    }
                    row_data.append(cell_data)
                else:
                    row_data.append({'text': '', 'value': None, 'name': '', 'row': row, 'original_row': row, 'is_numeric': False})
            
            batch_data.append(row_data)
        return batch_data


class LoadFileWorker(QThread):
    """Simple worker thread for loading files."""
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self._is_interruption_requested = False
    
    def requestInterruption(self):
        self._is_interruption_requested = True
    
    def run(self):
        try:
            if self._is_interruption_requested: return
            self.progress.emit("Loading file...")
            if Streamer is None:
                raise ImportError("SBC binary format reader not available")
            
            if self._is_interruption_requested: return
            streamer = Streamer(self.file_path)
            if self._is_interruption_requested: return
            data_dict = streamer.to_dict()
            if self.isInterruptionRequested(): return
            self.finished.emit(data_dict)
        except Exception as e:
            if not self._is_interruption_requested:
                self.error.emit(f"Error: {str(e)}")


class DataDialog(QDialog):
    """Simple dialog for viewing data details."""
    
    def __init__(self, name: str, data: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Data Viewer: {name}")
        self.resize(800, 600)
        
        # Store data (original_data is now primarily for stats view)
        self.original_data_for_stats = data 
        
        layout = QVBoxLayout()
        
        info = f"Variable: {name}\nType: {data.dtype}\nShape: {data.shape}\nSize: {data.size}"
        info_label = QLabel(info)
        info_label.setStyleSheet("background: #f8f9fa; padding: 10px; border: 1px solid #ddd;")
        layout.addWidget(info_label)
        
        # UI elements for TableViewManager
        self.array_table_widget = QTableWidget()
        self.array_progress_bar = QProgressBar()
        self.array_progress_label = QLabel("")
        self.array_load_all_btn = QPushButton("Load All Data")

        # Initialize TableViewManager for the array table
        self.table_manager = TableViewManager(
            table_widget=self.array_table_widget,
            progress_bar=self.array_progress_bar,
            progress_label=self.array_progress_label,
            load_all_button=self.array_load_all_btn,
            parent_widget=self
        )

        button_layout_for_array_table = QHBoxLayout()
        button_layout_for_array_table.addStretch()
        self.array_load_all_btn.setMaximumWidth(120)
        self.array_load_all_btn.setToolTip("Load all remaining array data at once")
        button_layout_for_array_table.addWidget(self.array_load_all_btn)
        layout.addLayout(button_layout_for_array_table)
        
        tabs = QTabWidget()
        
        if isinstance(data, np.ndarray) and data.size > 1:
            array_table_view_widget = QWidget()
            array_table_layout = QVBoxLayout()
            
            self.array_progress_bar.setVisible(False)
            self.array_progress_label.setVisible(False)
            self.array_load_all_btn.setVisible(False) # Initially hidden, manager controls it
            
            array_table_layout.addWidget(self.array_table_widget)
            array_table_layout.addWidget(self.array_progress_bar)
            array_table_layout.addWidget(self.array_progress_label)
            array_table_view_widget.setLayout(array_table_layout)
            
            # Configure the manager for this array data
            self.table_manager.configure_for_array(data)
            
            tabs.addTab(array_table_view_widget, "Table View")
        
        # Statistics/data view (uses self.original_data_for_stats)
        if self.original_data_for_stats.dtype.kind in ['i', 'u', 'f'] and self.original_data_for_stats.size > 0:
            text_stats = QTextEdit()
            text_stats.setReadOnly(True)
            text_stats.setFont(QFont("Courier", 10))
            
            stats = []
            stats.append(f"Min: {np.min(self.original_data_for_stats)}")
            stats.append(f"Max: {np.max(self.original_data_for_stats)}")
            stats.append(f"Mean: {np.mean(self.original_data_for_stats):.6f}")
            stats.append(f"Std: {np.std(self.original_data_for_stats):.6f}")
            
            if self.original_data_for_stats.size <= 1000:
                stats.append(f"\nData:\n{self.original_data_for_stats}")
            else:
                stats.append(f"\nFirst 20 values:\n{self.original_data_for_stats.flat[:20]}")
                stats.append(f"\nLast 20 values:\n{self.original_data_for_stats.flat[-20:]}")
            
            text_stats.setPlainText('\n'.join(stats))
            tabs.addTab(text_stats, "Statistics")
        else:
            text_raw = QTextEdit()
            text_raw.setReadOnly(True)
            text_raw.setFont(QFont("Courier", 10))
            
            if self.original_data_for_stats.size <= 100:
                text_raw.setPlainText(str(self.original_data_for_stats))
            else:
                text_raw.setPlainText(f"Large array with {self.original_data_for_stats.size} elements.\nFirst few: {self.original_data_for_stats.flat[:10]}")
            tabs.addTab(text_raw, "Raw Data")
        
        layout.addWidget(tabs)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        # Styles remain the same
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

    def closeEvent(self, event):
        """Clean up when dialog is closed."""
        try:
            if hasattr(self, 'table_manager'): # Ensure manager exists
                self.table_manager.close_gracefully()
            
            event.accept()
            
        except Exception as e:
            traceback.print_exc()
            event.accept() # Allow closing even if cleanup fails


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


class TableViewManager(QObject):
    """Manages a QTableWidget, data loading, and UI updates for table views."""

    def __init__(self, table_widget: QTableWidget, 
                 progress_bar: QProgressBar, progress_label: QLabel, 
                 load_all_button: QPushButton, parent_widget: QWidget):
        super().__init__(parent_widget)
        self.table = table_widget
        self.progress_bar = progress_bar
        self.progress_label = progress_label
        self.load_all_btn = load_all_button
        self.parent_widget = parent_widget

        self.original_data_source = None
        self.worker_type = None
        self.load_worker: BaseWorker | None = None
        
        self.loaded_rows = 0
        self.total_rows = 0
        self.batch_size = 1000
        self.large_batch_load_size = 5000

        self.is_loading = False
        self.load_all_mode = False
        self._sort_connected = False

        self.table.verticalScrollBar().valueChanged.connect(self._on_scroll)
        self.load_all_btn.clicked.connect(self._load_all_data_generic)

    def _get_worker_class(self):
        if self.worker_type == 'array':
            return ArrayLoadWorker
        elif self.worker_type == 'table':
            return TableLoadWorker
        return None

    def configure_for_array(self, data: np.ndarray):
        self.original_data_source = data
        self.worker_type = 'array'
        self.batch_size = 200 
        self.large_batch_load_size = 10000

        if data.ndim == 1:
            self.total_rows = data.size
            self.table.setColumnCount(1)
            self.table.setHorizontalHeaderLabels(["Value"])
        elif data.ndim == 2:
            self.total_rows = data.shape[0]
            max_cols = min(500, data.shape[1])
            self.table.setColumnCount(max_cols)
            self.table.setHorizontalHeaderLabels([f"{i}" for i in range(max_cols)])
        else:
            self.total_rows = data.flatten().size
            self.table.setColumnCount(1)
            self.table.setHorizontalHeaderLabels(["Value (Flattened)"])
        
        initial_rows_display = min(self.batch_size, self.total_rows)
        self.table.setRowCount(initial_rows_display)
        self.loaded_rows = 0
        self.table.setSortingEnabled(False)

        self.load_all_btn.setVisible(self.total_rows > self.batch_size)
        QTimer.singleShot(50, self._start_loading_batch)

    def configure_for_table(self, data_dict: Dict[str, Any]):
        self.original_data_source = data_dict
        self.worker_type = 'table'
        self.batch_size = 1000
        self.large_batch_load_size = 5000

        self.total_rows = max(len(data) for data in data_dict.values()) if data_dict else 0
        self.loaded_rows = 0

        if self.total_rows == 0:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            self.load_all_btn.setVisible(False)
            return

        initial_rows_display = min(self.batch_size, self.total_rows)
        self.table.setRowCount(initial_rows_display)
        self.table.setColumnCount(len(data_dict))
        self.table.setHorizontalHeaderLabels(list(data_dict.keys()))
        self.table.setSortingEnabled(False)

        self.load_all_btn.setVisible(self.total_rows > self.batch_size)
        QTimer.singleShot(50, self._start_loading_batch)

    def _start_loading_batch(self, is_large_batch_mode: bool = False):
        if self.loaded_rows >= self.total_rows and not is_large_batch_mode:
            return

        if self.is_loading:
            if not (is_large_batch_mode and self.load_all_mode):
                return

        self.is_loading = True
        WorkerClass = self._get_worker_class()
        if not WorkerClass:
            self.is_loading = False
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)
            if hasattr(self, 'load_all_mode') and self.load_all_mode:
                 self.load_all_mode = False
                 self.load_all_btn.setVisible(self.total_rows > 0 and self.loaded_rows < self.total_rows)
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_label.setVisible(True)
        
        current_batch_size = self.large_batch_load_size if is_large_batch_mode else self.batch_size
        self.progress_label.setText(f"Loading rows {self.loaded_rows} to {min(self.loaded_rows + current_batch_size, self.total_rows)}...")

        if self.worker_type == 'array':
            self.load_worker = WorkerClass(self.original_data_source, self.loaded_rows, current_batch_size)
        elif self.worker_type == 'table':
            self.load_worker = WorkerClass(self.original_data_source, self.loaded_rows, current_batch_size)
        else:
            self.is_loading = False
            return
        
        self.load_worker.setParent(None)
        self.load_worker.batch_ready.connect(self._on_batch_ready_generic)
        self.load_worker.progress.connect(self._on_loading_progress_generic)
        self.load_worker.finished.connect(
            self._on_large_batch_finished_generic if is_large_batch_mode else self._on_loading_finished_generic
        )
        self.load_worker.error.connect(self._on_loading_error_generic)
        # Store which finished slot was connected for precise disconnection later
        self.load_worker._connected_finish_slot = self._on_large_batch_finished_generic if is_large_batch_mode else self._on_loading_finished_generic

        if self.load_worker:
            self.load_worker.start()
        else:
            self.is_loading = False
            if self.load_all_mode:
                self.load_all_mode = False
                self.load_all_btn.setVisible(self.total_rows > 0 and self.loaded_rows < self.total_rows)
            self.progress_bar.setVisible(False)
            self.progress_label.setVisible(False)

    def _on_batch_ready_generic(self, start_row: int, batch_data: list):
        try:
            self.table.setUpdatesEnabled(False)
            required_rows = start_row + len(batch_data)
            if self.table.rowCount() < required_rows:
                self.table.setRowCount(required_rows)
            
            for i, row_items_data in enumerate(batch_data):
                table_row_idx = start_row + i
                for col_idx, cell_spec in enumerate(row_items_data):
                    text = cell_spec['text']
                    item = None
                    if self.worker_type == 'array':
                        item = QTableWidgetItem(text)
                        if cell_spec['is_numeric']:
                            try: item.setData(Qt.UserRole, float(cell_spec['value']))
                            except: pass
                        item.setData(Qt.UserRole + 1, cell_spec['original_index'])
                    elif self.worker_type == 'table':
                        if cell_spec['is_numeric'] and cell_spec['value'] is not None:
                            try: item = NumericTableWidgetItem(text, float(cell_spec['value']))
                            except: item = QTableWidgetItem(text)
                        else:
                            item = QTableWidgetItem(text)
                        if cell_spec['value'] is not None:
                            item.setData(Qt.UserRole, {
                                'value': cell_spec['value'], 'name': cell_spec['name'],
                                'row': cell_spec['row'], 'original_row': cell_spec['original_row']
                            })
                    
                    if item:
                        self.table.setItem(table_row_idx, col_idx, item)
            
            self.loaded_rows = start_row + len(batch_data)
            overall_progress_percent = int((self.loaded_rows / self.total_rows) * 100) if self.total_rows > 0 else 100
            self.progress_label.setText(f"Loaded {self.loaded_rows:,} of {self.total_rows:,} rows ({overall_progress_percent}%)")
            
            if self.load_all_mode: # If in load_all_mode, update bar with overall progress
                self.progress_bar.setValue(overall_progress_percent)
            # If not in load_all_mode, _on_loading_progress_generic handles the batch-specific progress bar.

            if self.worker_type == 'array':
                self._update_array_style_row_indices()
            
        except Exception as e:
            traceback.print_exc()
            self._on_loading_error_generic(f"Error displaying data: {str(e)}")
        finally:
            self.table.setUpdatesEnabled(True)

    def _on_loading_progress_generic(self, message: str, percentage: int):
        try:
            self.progress_label.setText(message) # Always update label with batch message
            if not self.load_all_mode: # Only update bar with batch % if NOT in load_all_mode
                self.progress_bar.setValue(percentage)
        except Exception as e:
            pass # Silently ignore if UI elements are gone

    def _on_loading_finished_generic(self):
        sender_worker = self.sender()
        current_active_worker = self.load_worker
        if not current_active_worker or sender_worker != current_active_worker:
            return
        try:
            all_data_loaded = self.loaded_rows >= self.total_rows
            if all_data_loaded:
                self._update_row_indices_after_sort()
            self.table.setSortingEnabled(True)
            if not self._sort_connected:
                self.table.horizontalHeader().sortIndicatorChanged.connect(self._update_row_indices_after_sort)
                self._sort_connected = True
        except Exception as e:
            traceback.print_exc()
            self._cleanup_worker_and_reset_state(current_active_worker, (self.loaded_rows >= self.total_rows), is_error=True)
            return
        self._cleanup_worker_and_reset_state(current_active_worker, (self.loaded_rows >= self.total_rows))

    def _on_large_batch_finished_generic(self):
        sender_worker = self.sender()
        current_active_worker = self.load_worker
        if not current_active_worker or sender_worker != current_active_worker:
            return
        try:
            if not (hasattr(self, 'load_all_mode') and self.load_all_mode):
                self._cleanup_worker_and_reset_state(current_active_worker, (self.loaded_rows >= self.total_rows), is_error=True)
                return

            if self.loaded_rows >= self.total_rows:
                self.progress_label.setText("All data loaded.") 
                QTimer.singleShot(100, lambda: self.progress_label.setVisible(False))
                try:
                    self.table.setSortingEnabled(True)
                    if not self._sort_connected:
                         self.table.horizontalHeader().sortIndicatorChanged.connect(self._update_row_indices_after_sort)
                         self._sort_connected = True
                    self._update_row_indices_after_sort()
                except Exception as e_inner:
                    traceback.print_exc()
                self._cleanup_worker_and_reset_state(current_active_worker, True)
            else:
                self._cleanup_worker_instance(current_active_worker)
                if self.load_worker == current_active_worker: 
                    self.load_worker = None 
                QTimer.singleShot(10, lambda: self._start_loading_batch(is_large_batch_mode=True))
        except Exception as e:
            traceback.print_exc()
            self._cleanup_worker_and_reset_state(current_active_worker, (self.loaded_rows >= self.total_rows), is_error=True)

    def _on_loading_error_generic(self, error_message: str):
        sender_worker = self.sender()
        current_active_worker = self.load_worker
        if current_active_worker and sender_worker != current_active_worker:
            if isinstance(sender_worker, BaseWorker):
                 self._cleanup_worker_instance(sender_worker)
            return
        QMessageBox.warning(self.parent_widget, "Loading Error", error_message)
        self._cleanup_worker_and_reset_state(current_active_worker, (self.loaded_rows >= self.total_rows), is_error=True)

    def _on_scroll(self, value):
        try:
            scroll_bar = self.table.verticalScrollBar()
            is_at_bottom = value >= scroll_bar.maximum() and scroll_bar.maximum() > 0
            has_more_data = self.loaded_rows < self.total_rows
            if is_at_bottom and has_more_data and not self.load_all_mode and not self.is_loading:
                self._start_loading_batch()
        except Exception as e:
            pass

    def _update_row_indices_after_sort(self):
        if self.worker_type == 'array':
            self._update_array_style_row_indices()
        elif self.worker_type == 'table':
            self._update_table_style_row_indices()

    def _update_array_style_row_indices(self):
        try:
            num_rows_to_label = self.table.rowCount()
            original_indices = []
            for r in range(num_rows_to_label):
                item = self.table.item(r, 0)
                if item:
                    original_idx = item.data(Qt.UserRole + 1)
                    original_indices.append(str(original_idx) if original_idx is not None else str(r))
                else:
                    original_indices.append(str(r))
            if self.loaded_rows < self.total_rows and num_rows_to_label < self.total_rows:
                 original_indices.extend([str(i) for i in range(num_rows_to_label, self.total_rows)])
            self.table.setVerticalHeaderLabels(original_indices)
        except Exception as e:
            pass

    def _update_table_style_row_indices(self):
        try:
            row_count = self.table.rowCount()
            if row_count == 0: return
            original_indices = []
            for r_idx in range(row_count):
                item = self.table.item(r_idx, 0)
                if item and item.data(Qt.UserRole):
                    data = item.data(Qt.UserRole)
                    if isinstance(data, dict) and 'original_row' in data:
                        original_indices.append(str(data['original_row']))
                    else: original_indices.append(str(r_idx))
                else: original_indices.append(str(r_idx))
            self.table.setVerticalHeaderLabels(original_indices)
        except Exception as e:
            pass

    def _load_all_data_generic(self):
        if self.loaded_rows >= self.total_rows or self.is_loading:
            return
        self.is_loading = True
        self.load_all_mode = True
        self.load_all_btn.setVisible(False)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        # Set initial overall progress for the bar
        current_overall_progress = int((self.loaded_rows / self.total_rows) * 100) if self.total_rows > 0 else 0
        self.progress_bar.setValue(current_overall_progress)
        
        self.progress_label.setVisible(True)
        # Set initial label for load all operation
        self.progress_label.setText(f"Loading all data... ({self.loaded_rows:,}/{self.total_rows:,} loaded)")
        
        if self.total_rows == 0:
            self.is_loading = False
            self.load_all_mode = False
            self.progress_bar.setVisible(False)
            self.progress_label.setText("No data to load.")
            QTimer.singleShot(1500, lambda: self.progress_label.setVisible(False))
            self.load_all_btn.setVisible(False)
            return
        self._start_loading_batch(is_large_batch_mode=True)

    def clear_data_and_worker(self):
        active_worker_to_stop = self.load_worker
        if active_worker_to_stop and active_worker_to_stop.isRunning():
            active_worker_to_stop.requestInterruption()
            wait_ms = 5000 if self.load_all_mode else 3000
            if not active_worker_to_stop.wait(wait_ms):
                print(f"TableViewManager: Worker [{getattr(active_worker_to_stop, 'worker_id', 'Unknown')}] did not terminate in {wait_ms}ms during clear_data.")
        self._cleanup_worker_instance(active_worker_to_stop)
        if self.load_worker == active_worker_to_stop:
            self.load_worker = None
        self.is_loading = False
        self.load_all_mode = False
        self.table.clearContents()
        self.table.setRowCount(0)
        self.original_data_source = None
        self.loaded_rows = 0
        self.total_rows = 0
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.load_all_btn.setVisible(False)

    def close_gracefully(self):
        active_worker_to_stop = self.load_worker
        if active_worker_to_stop and active_worker_to_stop.isRunning():
            active_worker_to_stop.requestInterruption()
            wait_time = 5000 if self.load_all_mode else 3000
            if not active_worker_to_stop.wait(wait_time): 
                print(f"TableViewManager: Worker [{getattr(active_worker_to_stop, 'worker_id', 'Unknown')}] thread did not terminate in {wait_time}ms for close.")
        self._cleanup_worker_instance(active_worker_to_stop)
        if self.load_worker == active_worker_to_stop:
            self.load_worker = None
        self.is_loading = False
        self.load_all_mode = False

    def _cleanup_worker_instance(self, worker_to_cleanup: BaseWorker | None):
        if not worker_to_cleanup:
            return
        try:
            # Targeted disconnection for the 'finished' signal
            if hasattr(worker_to_cleanup, '_connected_finish_slot'):
                try:
                    worker_to_cleanup.finished.disconnect(worker_to_cleanup._connected_finish_slot)
                except (RuntimeError, TypeError):
                    pass # Slot might have already been disconnected or was never connected properly
                try:
                    delattr(worker_to_cleanup, '_connected_finish_slot') # Clean up the attribute
                except AttributeError:
                    pass # Attribute wasn't there, which is fine

            # Disconnect other signals as before
            other_signals_to_disconnect = [
                (worker_to_cleanup.batch_ready, self._on_batch_ready_generic),
                (worker_to_cleanup.progress, self._on_loading_progress_generic),
                (worker_to_cleanup.error, self._on_loading_error_generic)
            ]
            for sig, slot in other_signals_to_disconnect:
                try:
                    sig.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass # Ignore if not connected or already disconnected

        except Exception as e_disc:
            # This broad exception is a fallback, but specific errors above should be handled.
            #print(f"Broad exception during worker cleanup: {e_disc}") 
            pass 
        worker_to_cleanup.deleteLater()

    def _cleanup_worker_and_reset_state(self, worker_just_finished: BaseWorker | None, all_data_actually_loaded: bool, is_error: bool = False):
        if worker_just_finished:
            self._cleanup_worker_instance(worker_just_finished)
        if self.load_worker == worker_just_finished:
            self.load_worker = None
        self.is_loading = False
        if (is_error and self.load_all_mode) or (not is_error and all_data_actually_loaded and self.load_all_mode):
            self.load_all_mode = False
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        if self.total_rows > 0 and self.loaded_rows < self.total_rows and not self.load_all_mode:
            self.load_all_btn.setVisible(True)
        else:
            self.load_all_btn.setVisible(False)


class DataViewer(QWidget):
    """Main data viewer widget."""
    
    def __init__(self):
        super().__init__()
        self.data_dict = None # Still used for tree and text views
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        # Tree view (remains unchanged by TableViewManager for now)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Variable", "Type", "Shape", "Preview"])
        self.tree.itemDoubleClicked.connect(self.on_tree_double_click)
        self.tabs.addTab(self.tree, "Tree View")
        
        # Table view widget setup
        table_view_container_widget = QWidget()
        table_layout = QVBoxLayout()
        
        # UI elements for TableViewManager for the main table
        self.main_table_widget = QTableWidget()
        self.main_table_progress_bar = QProgressBar()
        self.main_table_progress_label = QLabel("")
        self.main_table_load_all_btn = QPushButton("Load All Data")

        # Initialize TableViewManager for the main table
        self.table_manager = TableViewManager(
            table_widget=self.main_table_widget,
            progress_bar=self.main_table_progress_bar,
            progress_label=self.main_table_progress_label,
            load_all_button=self.main_table_load_all_btn,
            parent_widget=self
        )

        table_header_layout = QHBoxLayout()
        table_header_layout.addStretch()
        self.main_table_load_all_btn.setMaximumWidth(120)
        self.main_table_load_all_btn.setToolTip("Load all remaining data at once")
        self.main_table_load_all_btn.setVisible(False) # Manager controls visibility
        table_header_layout.addWidget(self.main_table_load_all_btn)
        
        self.main_table_widget.setSortingEnabled(True) # Initial state, manager may change it
        
        table_layout.addLayout(table_header_layout)
        table_layout.addWidget(self.main_table_widget)
        # Progress bar and label for the main table are now part of the overall layout below tabs
        table_view_container_widget.setLayout(table_layout)
        self.tabs.addTab(table_view_container_widget, "Table View")
        
        # Connect itemDoubleClicked for the main table directly
        self.main_table_widget.itemDoubleClicked.connect(self.on_table_double_click)

        # Text view (remains unchanged)
        self.text_view_widget = QTextEdit()
        self.text_view_widget.setReadOnly(True)
        self.text_view_widget.setFont(QFont("Courier", 10))
        self.tabs.addTab(self.text_view_widget, "Text View")
        
        # Overall layout for DataViewer (including progress for main table)
        layout.addWidget(self.tabs)
        layout.addWidget(self.main_table_progress_bar) # For the main table
        layout.addWidget(self.main_table_progress_label) # For the main table
        self.main_table_progress_bar.setVisible(False) # Manager controls visibility
        self.main_table_progress_label.setVisible(False) # Manager controls visibility
        self.setLayout(layout)
    
    def set_data(self, data_dict: Dict[str, Any]):
        self.data_dict = data_dict # Keep for tree and text views
        self.populate_tree() # Uses self.data_dict
        
        # Configure TableViewManager for the main table data
        if hasattr(self, 'table_manager'):
            self.table_manager.configure_for_table(data_dict)
        
        self.populate_text() # Uses self.data_dict
    
    def populate_tree(self):
        self.tree.clear()
        self.tree.setSortingEnabled(True)
        if not self.data_dict: return
        for name, data in self.data_dict.items():
            item = QTreeWidgetItem()
            item.setText(0, name)
            item.setText(1, str(data.dtype))
            item.setText(2, str(data.shape))
            item.setText(3, self._create_preview(data))
            item.setData(0, Qt.UserRole, data)
            self.tree.addTopLevelItem(item)
        for i in range(4):
            self.tree.resizeColumnToContents(i)
    
    def _create_preview(self, data: np.ndarray) -> str:
        if data.size == 0: return "Empty"
        if data.size == 1: return str(data.item())
        if data.size <= 5:
            return str(data.flat[:data.size].tolist())
        first_elements = ', '.join(str(data.flat[i]) for i in range(min(3, data.size)))
        return f"[{first_elements}, ...]"

    def populate_text(self):
        if not self.data_dict: 
            self.text_view_widget.clear()
            return
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
        self.text_view_widget.setPlainText('\n'.join(lines))
    
    def on_tree_double_click(self, item, column):
        data = item.data(0, Qt.UserRole)
        if data is not None and isinstance(data, np.ndarray):
            # Pass self (DataViewer) as parent for the dialog
            dialog = DataDialog(item.text(0), data, self) 
            dialog.show()
    
    def on_table_double_click(self, item: QTableWidgetItem):
        # This is for the main table, managed by self.table_manager
        # The item's UserRole data should be set by TableViewManager._on_batch_ready_generic
        item_data = item.data(Qt.UserRole)
        if item_data and isinstance(item_data, dict) and 'value' in item_data and isinstance(item_data['value'], np.ndarray):
            # Ensure parent is correctly passed if DataDialog needs it (e.g., self or self.window())
            dialog = DataDialog(f"{item_data.get('name','Array')}[{item_data.get('row','-')}]", item_data['value'], self) 
            dialog.show()
    
    def clear_data(self):
        # Clear data for tree and text views
        self.tree.clear()
        self.text_view_widget.clear()
        self.data_dict = None 
        
        # Clear data managed by TableViewManager for the main table
        if hasattr(self, 'table_manager'):
            self.table_manager.clear_data_and_worker()
        
        # Reset sort connection flag (if it was specific to DataViewer internal table management)
        # self._sort_connected = False # This flag is now internal to TableViewManager

    def get_table_manager_for_cleanup(self) -> TableViewManager | None:
        """Provides access to the table manager for external cleanup if needed."""
        if hasattr(self, 'table_manager'):
            return self.table_manager
        return None

class SBCViewer(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.file_load_worker = None # Renamed for clarity from self.load_worker
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
            self.file_load_worker = LoadFileWorker(file_path)
            self.file_load_worker.setParent(None)
            self.file_load_worker.finished.connect(self.on_loaded)
            self.file_load_worker.error.connect(self.on_error)
            self.file_load_worker.progress.connect(self.statusBar().showMessage)
            self.file_load_worker.start()
    
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

    def closeEvent(self, event):
        """Handle window close event to clean up resources."""
        try:
            if hasattr(self, 'viewer') and self.viewer:
                table_manager = self.viewer.get_table_manager_for_cleanup()
                if table_manager: table_manager.close_gracefully()
            
            if self.file_load_worker and self.file_load_worker.isRunning():
                self.file_load_worker.requestInterruption()
                if not self.file_load_worker.wait(3000):
                    print("SBCViewer: FileLoadWorker did not terminate in 3000ms after requesting interruption.")
            
            if self.file_load_worker: self.file_load_worker.deleteLater(); self.file_load_worker = None
            
            event.accept()

        except Exception as e:
            traceback.print_exc()
            event.accept() # Allow closing even if cleanup fails


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