import numpy as np
import cv2
from threading import Thread
# Cola import
from ColaLib import numpy_image_to_pixmap
# Qt import
from Ui_RoiWindowDialog import Ui_Dialog
from PyQt5.QtWidgets import (QWidget, QLabel, QTableWidgetItem)
from PyQt5.QtCore import pyqtSignal, QTimer, pyqtSignal
from PyQt5.QtGui import QMouseEvent, QImage, QBrush, QColor, QIcon



class ColaClickLable(QLabel):
    button_clicked_signal = pyqtSignal(QMouseEvent)
    
    def __init__(self, parent=None):
        super(ColaClickLable, self).__init__(parent)
        
    
    def mouseReleaseEvent(self, mouse_event):
        self.button_clicked_signal.emit(mouse_event)
        
        
    def connect_customized_slot(self, func):
        self.button_clicked_signal.connect(func)




class ColaRoiWindow(QWidget, Ui_Dialog):
    my_finish_signal = pyqtSignal(object)
    my_cancel_signal = pyqtSignal(object)
    
    my_any_signal = pyqtSignal(object) 
    """
        int(0): reload red and orange model
    """
    
    def __init__(self, parent, cola_camera, coord_faces):
        super(ColaRoiWindow, self).__init__()
        self.setupUi(self)
        self.parent = parent
        self.coord_faces = coord_faces
        self.curr_num = int(0)
        self.curr_idx_face = int(0)
        self.curr_row = int(0)
        self.curr_col = int(0)
        self.label2_img = ColaClickLable(self)
        self.X = 0
        self.Y = 0
        self.cola_camera = cola_camera
        self.is_showing_camera = True
        self.timer_show_img = QTimer()
        self.img = None
        self.dialog_state = int(1)  # 0: roi 1: sample default
        self.color_sample_list = []
        self.color_to_sample = None
        
        # init signal and slot
        self.init_singal_slot()
        self.init_ui()
        
    
    def init_ui(self):
        # default curr sample color 
        self.bbox_color.setCurrentIndex(0)

        #         
        self.label2_img.setGeometry(480, 100, 600, 600)
        self.label2_img.setText("图片显示区域")

        # window set
        self.setWindowTitle("Set Roi")
        self.setWindowIcon(QIcon("./cola_store/cube_dialog.jpg"))
        
        # init table
        self.cola_init_show_coord_table()
        
        # pbtn background
        self.pbtn_next.setStyleSheet(
            '''QPushButton{background:#a8ffbc;border-radius:5px;}QPushButton:hover{background:#e6402e;}'''
        )
        self.pbtn_pre.setStyleSheet(
            '''QPushButton{background:#a8ffbc;border-radius:5px;}QPushButton:hover{background:#e6402e;}'''
        )
        self.pbtn_finish.setStyleSheet(
            '''QPushButton{background:#a8ffbc;border-radius:5px;}QPushButton:hover{background:#e6402e;}'''
        )
        self.pbtn_cancel.setStyleSheet(
            '''QPushButton{background:#a8ffbc;border-radius:5px;}QPushButton:hover{background:#e6402e;}'''
        )
        self.pbtn_state_change.setStyleSheet(
            '''QPushButton{background:#00a8f3;border-radius:5px;}'''    
        )
        
        # init ledit hsv
        self.ledit_hsv.setFixedSize(150, 28)
        self.ledit_bgr.setFixedSize(150, 28)
        self.ledit_color.setFixedSize(150, 28)
        
    
    def init_singal_slot(self):
        self.timer_show_img.timeout.connect(self.cola_slot_show_label_img)
        self.timer_show_img.start(1000)

        # For face name
        self.bbox_facename.currentTextChanged.connect(self.cola_slot_update_face)
        
        # next or pre point
        self.pbtn_next.clicked.connect(self.cola_slot_next_point)
        self.pbtn_pre.clicked.connect(self.cola_slot_pre_point)
        
        # finish pushbottom
        self.pbtn_finish.clicked.connect(self.cola_slot_finish)
        self.pbtn_cancel.clicked.connect(self.cola_slot_cancel)
        
        # img
        self.label2_img.connect_customized_slot(self.cola_slot_show_coord_in_table)
        
        # state change
        self.pbtn_state_change.clicked.connect(self.cola_slot_state_change)

        # sample state
        self.pbtn_resample.clicked.connect(self.cola_slot_pbtn_resample_change)
        self.pbtn_store_color.clicked.connect(self.cola_slot_pbtn_store_color)
        self.pbtn_reload_model.clicked.connect(self.cola_slot_pbtn_reload_model)
        self.bbox_color.currentTextChanged.connect(self.cola_slot_update_sample_color)
        
        
    def cola_slot_update_sample_color(self):
        self.color_sample_list = []
        self.color_to_sample = self.bbox_color.currentText()

    
    def cola_slot_pbtn_reload_model(self):
        """
        brief: Reload the classifier of red and orange using latest color data
        """
        print("cola_slot_pbtn_reload_model: begin")    
        self.my_any_signal.emit(int(0)) 
        print("cola_slot_pbtn_reload_model: end")    
        
        
    
    def cola_slot_pbtn_store_color(self):
        """
        brief: store color and trans to hsv data
        note: red and orange will recover, other color will append
        """
        print("cola_slot_pbtn_store_color: begin")
        self.color_to_sample = self.bbox_color.currentText()
        filename = "./cola_store/color_mat_{}.txt".format(self.color_to_sample)
        print(filename)
        new_data = np.asarray(self.color_sample_list, dtype=np.ubyte)
        old_data = np.loadtxt(filename).astype(np.ubyte)
        new_data = np.vstack((old_data, new_data))
        np.savetxt(filename, new_data, fmt='%2d')
        
        # trans color to hsv and save
        from ColaExtraColor import trans_bgr_to_hsv
        trans_bgr_to_hsv([self.color_to_sample])
        print("cola_slot_pbtn_store_color: end")
        
        
    def cola_slot_pbtn_resample_change(self):
        self.color_sample_list = []
        
    
    
    def cola_slot_state_change(self):
        """
        Change self.state, init sample about tool
        """
        # init variable
        self.cola_slot_update_sample_color()
        
        # state change
        if self.dialog_state == 0:  # set to sample mode
            self.pbtn_state_change.setText("采样模式")
            self.pbtn_state_change.setStyleSheet(
                '''QPushButton{background:#00a8f3;border-radius:5px;}'''    
            )
            self.dialog_state = 1
        else:  # set to roi mode
            self.pbtn_state_change.setText("标定模式")
            self.pbtn_state_change.setStyleSheet(
                '''QPushButton{background:#fff200;border-radius:5px;}'''    
            )
            self.dialog_state = 0
    
    def close(self):
        self.parent.cola_extra_color.save_parameters()
    
        
    def cola_slot_finish(self):
        self.my_finish_signal.emit(self.coord_faces)
        self.close()
        
        
    def cola_slot_cancel(self):
        self.my_cancel_signal.emit(None)
        self.close()
        
        
    def cola_slot_pre_point(self):
        self.pre_point()
        
        
    def cola_slot_next_point(self):
        self.next_point()
        
    
    def cola_slot_update_face(self):
        self.update_idx_face()
        self.curr_num = 0
        self.update_point()
        self.cola_init_show_coord_table()
 
    
    def cola_init_show_coord_table(self):
        for row in range(3):
            for col in range(3):
                info = "({}, {})".format(
                    self.coord_faces[self.curr_idx_face, row, col, 0], 
                    self.coord_faces[self.curr_idx_face, row, col, 1])
                item = QTableWidgetItem(info)
                self.table_coord.setItem(row, col, item)
                self.table_coord.item(row, col).setBackground(
                    QBrush(QColor(255, 255, 255))    
                )
        self.table_coord.item(0, 0).setBackground(
            QBrush(QColor(234, 227, 99))
        )
    
    
    my_color_change_singal = pyqtSignal(object)
    def cola_slot_show_coord_in_table(self, mouse_event):
        pos = mouse_event.pos()
        # update table and X and Y
        self.X = pos.x()
        self.Y = pos.y()

        # roi state
        if self.dialog_state == 0:  # roi
            # update coord tensor
            self.coord_faces[self.curr_idx_face, self.curr_row, self.curr_col, 0] = self.X
            self.coord_faces[self.curr_idx_face, self.curr_row, self.curr_col, 1] = self.Y
            
            # update table item
            info = "({}, {})".format(self.X, self.Y)
            item = QTableWidgetItem(info)
            self.table_coord.setItem(self.curr_row, self.curr_col, item)

            # white table
            self.next_point()

            # update x y line edit
            self.ledit_x.setText(str(self.X))
            self.ledit_y.setText(str(self.Y))
        elif self.dialog_state == 1: # sample: color
            # sample color to list
            color = self.img[pos.y(), pos.x()]
            self.color_sample_list.append(color)
        
        # update ledit color
        self.my_color_change_singal.emit(self.img[pos.y(), pos.x()])
        
    
    def cola_slot_show_label_img(self):
        img = self.cola_camera.get_face_image(self.curr_idx_face)
        self.img = img.copy()
        self.parent.cola_extra_color._draw_roi(img, self.curr_idx_face)
        # self.img = self.parent.cola_extra_color.get_idx_face_img_with_roi(self.curr_idx_face)
        self.label2_img.setPixmap(numpy_image_to_pixmap(img))
            
    
    def update_idx_face(self):
        text = self.bbox_facename.currentText()
        self.curr_idx_face = int(text[0])
    
    
    def next_point(self):
        self.table_coord.item(self.curr_row, self.curr_col).setBackground(
            QBrush(QColor(255, 255, 255))
        )
        self.curr_num += 1
        self.update_point()
        self.table_coord.item(self.curr_row, self.curr_col).setBackground(
            QBrush(QColor(234, 227, 99))
        )
    
    def pre_point(self):
        self.table_coord.item(self.curr_row, self.curr_col).setBackground(
            QBrush(QColor(255, 255, 255))
        )
        self.curr_num -= 1
        self.update_point()
        self.table_coord.item(self.curr_row, self.curr_col).setBackground(
            QBrush(QColor(234, 227, 99))
        )
        
    
    def update_point(self):
        if self.curr_num >= 9:
            self.curr_num = 0
        elif self.curr_num <= -1:
            self.curr_num = 8
        self.curr_row = int(self.curr_num / 3)
        self.curr_col = self.curr_num % 3
    
    def run(self, cola_camera):
        self.show()
        return self.coord_faces
        
    
    
        
        
