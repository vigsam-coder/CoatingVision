import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                             QSlider, QMessageBox, QSizePolicy, QGridLayout)
from PyQt5.QtCore import Qt, QDir, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QCursor


class SegmentationQCApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Segmentation Label QC")
        # Get screen size
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()

        # Calculate width and height 
        width = int(screen_geometry.width() * 0.9)  # 90% of the screen width
        height = int(screen_geometry.height() * 0.9)  # 90% of the screen height

        # Center the window
        x = (screen_geometry.width() - width) // 2
        y = (screen_geometry.height() - height) // 2

        # Set geometry dynamically
        self.setGeometry(x, y, width, height)

        self.initUI()

        self.image_folder = None
        self.mask_folder = None
        self.image_files = []
        self.current_index = 0

    def initUI(self):
        # Overlay display for image and mask
        self.overlay_display = MaskEditor()
        self.overlay_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Buttons for actions
        load_button = QPushButton("Load Image & Mask Folders")
        load_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        load_button.clicked.connect(self.load_image_and_mask_folders)

        next_button = QPushButton("Next Image")
        next_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        next_button.clicked.connect(self.next_image)

        prev_button = QPushButton("Previous Image")
        prev_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        prev_button.clicked.connect(self.previous_image)

        save_button = QPushButton("Save Mask")
        save_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        save_button.clicked.connect(self.save_mask)

        # Brush size slider
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(50)
        self.brush_slider.setValue(10)
        self.brush_slider.valueChanged.connect(self.change_brush_size)

        # Brush and Eraser buttons
        brush_button = QPushButton("Brush Tool")
        brush_button.clicked.connect(self.set_brush_tool)

        eraser_button = QPushButton("Eraser Tool")
        eraser_button.clicked.connect(self.set_eraser_tool)

        main_layout = QVBoxLayout()

        # Add overlay display to the main layout
        main_layout.addWidget(self.overlay_display)

        # Control layout for buttons in a grid
        control_layout = QGridLayout()
        control_layout.addWidget(load_button, 0, 0)  # Row 0, Column 0
        control_layout.addWidget(save_button, 0, 1)  # Row 0, Column 1
        control_layout.addWidget(prev_button, 1, 0)  # Row 1, Column 0
        control_layout.addWidget(next_button, 1, 1)  # Row 1, Column 1
        control_layout.addWidget(QLabel("Brush Size"), 2, 0)  # Below buttons
        control_layout.addWidget(self.brush_slider, 2, 1)  # Below buttons
        control_layout.addWidget(brush_button, 3, 0)  # Below brush size
        control_layout.addWidget(eraser_button, 3, 1)  # Below brush size

        # Add the control layout to the main layout
        main_layout.addLayout(control_layout)

        # Set the central widget with the main layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_image_and_mask_folders(self):
        # Load image and mask folders using QFileDialog
        self.image_folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        self.mask_folder = QFileDialog.getExistingDirectory(self, "Select Mask Folder")

        if self.image_folder and self.mask_folder:
            self.image_files = sorted(os.listdir(self.image_folder))
            if self.image_files:
                # Load the first image and mask
                first_image_filename = self.image_files[0]
                self.load_image_and_mask(first_image_filename)

    def load_image_and_mask(self, filename):
        # Load the current image and corresponding mask
        image_path = os.path.join(self.image_folder, filename)


        base_filename = os.path.splitext(filename)[0]  # Get filename without extension
        mask_path = os.path.join(self.mask_folder, base_filename + '.png')

        # Check if both the image and mask files exist
        if os.path.exists(image_path) and os.path.exists(mask_path):
            # Load image and mask into the overlay editor
            self.overlay_display.load_image_and_mask(image_path, mask_path)
        else:
            QMessageBox.warning(self, "Error", f"Image or mask not found for {filename}.")

    def next_image(self):
        if self.image_files:
            self.current_index = (self.current_index + 1) % len(self.image_files)
            self.load_image_and_mask(self.image_files[self.current_index])

    def previous_image(self):
        if self.image_files:
            self.current_index = (self.current_index - 1) % len(self.image_files)
            self.load_image_and_mask(self.image_files[self.current_index])

    def save_mask(self):
        if self.image_files:
            current_file = self.image_files[self.current_index]
            save_path = os.path.join(self.mask_folder, current_file)
            self.overlay_display.save_mask(save_path)

    def change_brush_size(self):
        self.overlay_display.set_brush_size(self.brush_slider.value())

    def set_brush_tool(self):
        self.overlay_display.set_tool("brush")

    def set_eraser_tool(self):
        self.overlay_display.set_tool("eraser")


class MaskEditor(QLabel):
    def __init__(self):
        super().__init__()
        self.image = None
        self.mask = None
        self.drawing = False
        self.pen_size = 10
        self.tool = "brush"  # Default tool
        self.setCursor(self.create_round_cursor(self.pen_size))

    def create_round_cursor(self, size):
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setBrush(QColor(255, 255, 255, 255))
        painter.drawEllipse(0, 0, size, size)
        painter.end()
        return QCursor(pixmap)

    def load_image_and_mask(self, image_path, mask_path):
        self.image = QImage(image_path)
        self.mask = QImage(mask_path).convertToFormat(QImage.Format_ARGB32)
        self.update()

    def map_cursor_to_image(self, cursor_pos):
        if self.image is not None:
            return cursor_pos
        return QPoint(cursor_pos.x(), cursor_pos.y())

    def save_mask(self, save_path):
        if self.mask:
            self.mask.save(save_path.replace('.jpg','.png'))

    def set_tool(self, tool_type):
        self.tool = tool_type

    def set_brush_size(self, size):
        self.pen_size = size
        self.setCursor(self.create_round_cursor(size))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.image and self.mask:
            mapped_point = self.map_cursor_to_image(event.pos())
            self.drawing = True
            self.last_point = mapped_point

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.drawing and self.image and self.mask:
            mapped_point = self.map_cursor_to_image(event.pos())
            painter = QPainter(self.mask)
            pen = QPen(QColor(255, 255, 255, 255), self.pen_size, Qt.SolidLine, Qt.RoundCap,
                       Qt.RoundJoin) if self.tool == "brush" else QPen(QColor(0, 0, 0, 255), self.pen_size,
                                                                       Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, mapped_point)
            self.last_point = mapped_point
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.image and self.mask:
            self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)

        if self.image:
            painter.drawImage(0, 0, self.image)

        if self.mask:
            painter.setOpacity(0.4)  # Set mask transparency
            painter.drawImage(0, 0, self.mask)
            painter.setOpacity(1.0)  # Reset opacity

        painter.end()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = SegmentationQCApp()
    mainWin.show()
    sys.exit(app.exec_())
