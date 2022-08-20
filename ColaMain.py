import sys
import cv2
from threading import Thread, Semaphore
import time
import random
import numpy as np
import argparse
import winsound
from enum import Enum
import os
# Cola import
from ColaCamera import ColaCamera
from ColaCluster import ColaCluster
from ColaSocket import ColaSocket
from ColaExtraColor import ColaExtraColor
import ColaSolver
from ColaCamera import show_all_cammera
from ColaLib import numpy_image_to_pixmap
# Qt import
from Ui_RubiksRobot import Ui_MainWindow
from ColaRoiDialog import ColaRoiWindow
from PyQt5.QtWidgets import (
    QApplication, 
    QMainWindow, 
    QToolTip,
    QPushButton,
    QMessageBox,
    QTextEdit,
    QDesktopWidget,
    QHBoxLayout,
    QVBoxLayout,
    QDialog,
    QWidget
)
from PyQt5.QtGui import QFont, QPixmap, QIcon, QImage, QPalette, QColor, QFont
from PyQt5.QtCore import QTimer, QThread, Qt, pyqtSignal



class ThreadForRunningState(QThread):
    def __init__(self, parent):
        super(ThreadForRunningState, self).__init__()
        self.parent = parent
    
    def run(self):
        self.parent.cola_runing_sample_solve()
        self.parent.cola_show_all_color_label(is_show_label=True)
        self.parent.is_runing_state = False
        
        
class ThreadForFlashRuningState(QThread):
    def __init__(self, pbtn_running, parent):
        super(ThreadForFlashRuningState, self).__init__()
        self.parent = parent
        self.pbtn_running = pbtn_running
    
    def run(self):
        while self.parent.is_runing_state:
            self.pbtn_running.setStyleSheet(
                '''QPushButton{background:#00a8f3;border-radius:5px;}'''
            )
            time.sleep(1)
            self.pbtn_running.setStyleSheet(
                '''QPushButton{background:#fff200;border-radius:5px;}'''
            )
            time.sleep(1)
        self.pbtn_running.setStyleSheet(
            '''QPushButton{background:#a8ffbc;border-radius:5px;}'''
        )


class ColaAnySingal(Enum):
    TEDIT_SEND_MSG = 1


class ColaMainWindow(QMainWindow, Ui_MainWindow):
    my_any_signal = pyqtSignal(ColaAnySingal)  # 0: update self.msg -> tedit_show
    
    def __init__(self, parent=None, is_test=True):
        super(ColaMainWindow, self).__init__(parent)

        self.setupUi(self)
        # variable
        self.flag_show_img = True
        self.timer_net_state = QTimer()
        self.color_tensor = np.zeros(shape=(6, 3, 3, 3), dtype=np.int8)
        self.label_tensor = np.zeros(shape=(6, 3, 3), dtype=np.int32)
        self.cola_cluster = ColaCluster()
        self.msg = ""
        self.solving_duration = 0
        self.cola_camera = ColaCamera(is_test=is_test)
        self.cola_extra_color = ColaExtraColor(self.cola_camera)
        self.dlg = ColaRoiWindow(self, self.cola_camera, self.cola_extra_color.coord_faces)  
        self.size_w = 1650
        self.size_h = 840
        self.id_cap = 0
        
        # for runing state
        self.is_runing_state = False  # is runing 
        self.is_solved = False
        self.runing_state_t = ThreadForRunningState(self)
        self.flash_bottom_t = ThreadForFlashRuningState(
            self.pbtn_running, 
            self)
        
        # About network
        IP = self.ledit_ipaddr.text()
        port = self.ledit_netport.text()
        self.cola_network = ColaSocket(IP, int(port))
        
        # init function
        self.initUI()
        self.connect_single_slot()

        # test
        self.coal_save_four_camera_img()
        
    
    def connect_single_slot(self):
        # timer for show network state
        # self.timer_net_state.timeout.connect(self.cola_show_net_state)
        # self.timer_net_state.start(1000)
        
        # connect for network
        self.pbtn_reconnect.clicked.connect(self.cola_slot_reconnect_network)

        # connect signal and slot
        self.pbtn_sample.clicked.connect(self.cola_slot_sample)
        self.pbtn_store_sample.clicked.connect(self.cola_slot_store_sample)
        self.pbtn_solve.clicked.connect(self.cola_slot_solve)
        self.pbtn_send.clicked.connect(self.cola_slot_send)
        self.pbtn_roi.clicked.connect(self.cola_slot_setroi)
        self.pbtn_recv.clicked.connect(self.cola_slot_resv)
        
        # roi dialog close
        self.dlg.my_finish_signal.connect(self.cola_slot_dlg_close)
        self.dlg.my_cancel_signal.connect(self.cola_slot_dlg_close)
        
        """ Disable the singal to change the camera state
        # cap id or rotation change
        self.bbox_capid.currentTextChanged.connect(self.cola_slot_bbox_capid_or_rotation_changed)
        self.bbox_rotation.currentTextChanged.connect(self.cola_slot_bbox_capid_or_rotation_changed)
        
        # slider change -> cap state parameter
        self.slider_sx.sliderMoved.connect(self.cola_slot_slider_sx_change)
        self.slider_sy.sliderMoved.connect(self.cola_slot_slider_sy_change)
        self.slider_width.sliderMoved.connect(self.cola_slot_slider_width_change)
        """
        
        # dlg color change -> predict and show in dlg
        self.dlg.my_color_change_singal.connect(self.cola_slot_dlg_color_change)
        
        # reload model signal
        self.dlg.my_any_signal.connect(self.cola_slot_dlg_reload_knn_model)
        
        # runing state
        self.pbtn_running.clicked.connect(self.cola_slot_runing_btn_clicked)
        
        # calculate length of message
        self.tedit_send_msg.textChanged.connect(self.cola_slot_steps_length)

        # for my any signal
        self.my_any_signal.connect(self.cola_slot_any_signal)
        
        
    def coal_save_four_camera_img(self):
        img_total = np.zeros(shape=(1200, 1200, 3), dtype=np.ubyte)
        for row in range(2):
            for col in range(2):
                img = self.cola_extra_color.img_with_roi(row * 2 + col)
                range_row = np.arange(600 * row, (row + 1) * 600)
                range_col = np.arange(600 * col, (col + 1) * 600)
                mesh_col, mesh_row = np.meshgrid(range_col, range_row)
                img_total[mesh_col, mesh_row] = img
        cv2.imwrite("./cola_store/img_tmp.png", img_total)


    def cola_slot_store_sample(self):
        num_list = [int(i[0:i.find('.')]) for i in os.listdir("./cola_color_store")]
        next_id = max(num_list) + 1
        filename = "./cola_color_store/{}.txt".format(next_id)
        np.savetxt(filename, self.color_tensor.reshape(-1, 3), fmt='%d')
    
    
    
    def cola_slot_any_signal(self, task):
        if task is ColaAnySingal.TEDIT_SEND_MSG:
            self.tedit_send_msg.setText(self.msg)
            # solving duration
            print(self.solving_duration)
            self.ledit_duration.setText("{:2f} (s)".format(self.solving_duration))
        
    
    def cola_slot_steps_length(self):
        # length
        self.label_steps_len.setText("STEPS: {}".format(len(self.tedit_send_msg.toPlainText())))
        
        # prediction duration
        duration_ms = ColaSolver.get_time_duration()
        if duration_ms > 1000:
            self.label_duration.setText("{:.2f}s".format(duration_ms / 1000.0))
        else:
            self.label_duration.setText("")
        
        
    def keyPressEvent(self, QKeyEvent):
        if QKeyEvent.key() == Qt.Key_Return:
            self.cola_slot_runing_btn_clicked()
        if QKeyEvent.key() == Qt.Key_C:
            self.cola_slot_reconnect_network()
    
    
    def cola_runing_state_flash_btn(self, num):
        while self.is_runing_state:
            self.pbtn_running.setStyleSheet(
                '''QPushButton{background:#00a8f3;border-radius:5px;}'''
            )
            time.sleep(1)
            self.pbtn_running.setStyleSheet(
                '''QPushButton{background:#fff200;border-radius:5px;}'''
            )
            time.sleep(1)
        self.pbtn_running.setStyleSheet(
            '''QPushButton{background:#a8ffbc;border-radius:5px;}'''
        )
        
    
    def check_label(self):
        """
        brief: check label, show info in ledit_show
        return: True if label right
        """
        for idx_color in range(6):
            if np.sum(self.label_tensor == idx_color) != 9:
                return False
        
        return True
    
    def cola_slot_runing_btn_clicked(self):
        self.pbtn_running.clicked.disconnect(self.cola_slot_runing_btn_clicked)
        winsound.PlaySound('SystemExclamation', winsound.SND_ASYNC)
        self.is_runing_state = not self.is_runing_state
        if self.is_runing_state:
            self.pbtn_running.setText("比赛模式")
            
            # set begin string of msg
            self.bbox_send_char.setCurrentIndex(0)
            
            # flash buttom
            # cola_t = Thread(target=(self.cola_runing_state_flash_btn))
            # cola_t.start()
            self.flash_bottom_t.start()
                    
            # Start rung version standard thread
            cola_t = Thread(target=self.cola_runing_state)
            cola_t.start()
            
            # version of QThread
            # self.runing_state_t.start()
        else:
            self.pbtn_running.setText("预备模式")
            
        self.pbtn_running.clicked.connect(self.cola_slot_runing_btn_clicked)
    
    
    def cola_runing_state(self):
        """
        brief: state for competing
        1. this thread: receive msg from stm32 and do the corresponding acton by this msg
        2. create a thread to constantly extract the color and label it than try to solve
        """
        self.cola_runing_sample_solve()
        self.cola_show_all_color_label(is_show_label=True)
        self.is_runing_state = False
        self.pbtn_running.setText("预备模式")
        

    def cola_runing_sample_solve(self):
        """
        brief: try solve, break if succeed of exit runing state, and set self.msg and tedit_send_msg
        """
        # count = 0
        while self.is_runing_state:
            # just test
            # time.sleep(0.01)
            # self.msg = '1234567890'
            # # self.tedit_send_msg.setText(str(count))
            # count += 1
            # print(count)
            # if count > 100:
            #     break
            # continue
            
            # real
            self.color_tensor = self.cola_extra_color.extra_color()
            t1 = time.time()
            self.label_tensor = self.cola_cluster.run(self.color_tensor)

            # check label
            # if not self.check_label():
                # continue

            # try to solve
            try:
                self.msg = ColaSolver.rubiks_solver(self.label_tensor)
                # 
                self.cola_network.send_msg(self.msg, self.bbox_send_char.currentText())
                winsound.Beep(4000, 2000)
                # send message
                # self.statusBar().showMessage("Solve Succeed", 1000)
                # bee
                # show in mainWindow
                # send message
                # save img 
                self.coal_save_four_camera_img()
                # self.msg = "34567890"
                self.solving_duration = time.time() - t1
                # self.msg = "12345678"
                self.my_any_signal.emit(ColaAnySingal.TEDIT_SEND_MSG)
            except Exception as ex:
                print(ex)
            else:
                print("Solved: {}".format(self.msg))
                break
        

        
    
    def cola_slot_disconnect_network(self):
        self.cola_network.disconnect()
        self.cola_say_new_line_ledit("Network", "Disconnected.")
    
    
    def cola_slot_reconnect_network(self):
        self.cola_network.reconnect(self.ledit_ipaddr.text(), int(self.ledit_netport.text()))
        self.cola_show_net_state()
        self.cola_say_new_line_ledit("Network", "Reconnected" if self.cola_network.is_connected else "Reconnected error." )
        pass
        
    
    def cola_slot_dlg_reload_knn_model(self, signal):
        if signal != 0:
            return
        self.cola_cluster.reload_knn_ro_model()
        
    
    def cola_slot_dlg_color_change(self, color):
        color_name_vec = ["Yellow", "Orange", "Green", "White", "Red", "Blue"]
        idx_color = self.cola_cluster.classify_hsv(color)
        self.dlg.ledit_color.setText(color_name_vec[idx_color])
        self.dlg.ledit_bgr.setText(str(color))
        color_hsv = cv2.cvtColor(color.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(3)
        self.dlg.ledit_hsv.setText(str(color_hsv))
        
        
    def cola_slot_slider_width_change(self):
        width = self.slider_width.value()
        self.cola_camera.cap_state[self.id_cap].width = width
        
        
    def cola_slot_slider_sy_change(self):
        sy = self.slider_sy.value()
        self.cola_camera.cap_state[self.id_cap].sy = sy
    
    def cola_slot_slider_sx_change(self):
        sx = self.slider_sx.value()
        self.cola_camera.cap_state[self.id_cap].sx = sx
        
    
    def cola_slot_bbox_capid_or_rotation_changed(self):
        """
        When capid changed, change the range of slider, init sx, sy, rotation
        """
        self.id_cap = int(self.bbox_capid.currentText()[0]) - 1
        sx = self.cola_camera.cap_state[self.id_cap].sx
        sy = self.cola_camera.cap_state[self.id_cap].sy
        width = self.cola_camera.cap_state[self.id_cap].width
        rotation = self.cola_camera.cap_state[self.id_cap].rotation
        
        # range of slider
        range_x, range_y = 480, 640
        width_max = 480 - sx
        if rotation == 90 or rotation == 270:
            range_x, range_y = 640, 480
            sx, sy = sy, sx  # todo: exchange?
            width_max = 640 - sx
        
        # show 
        print(
            "SX: ", sx, 
            "SY: ", sy, 
            "WIDTH: ", width, 
            "ROTATION: ", rotation, 
            "RANGE_X: ", range_x, 
            "RANGE_Y: ", range_y, 
            "WIDTH_MAX: ", width_max)    
        
        self.slider_sx.setMinimum(0)
        self.slider_sx.setMaximum(range_x)
        self.slider_sy.setMinimum(0)
        self.slider_sy.setMaximum(range_y)
        self.slider_width.setMinimum(0)
        self.slider_width.setMaximum(width_max)
        
        # init slider
        self.slider_sx.setValue(sx)
        self.slider_sy.setValue(sy)
        
        # init rotation, width
        self.bbox_rotation.setEditText(str(rotation))
        self.slider_width.setValue(width)
        
        
        
    
    def cola_slot_dlg_close(self, coords_face):
        result = coords_face
        if result is not None:
            self.cola_extra_color.coord_faces = result
        
    
    def cola_slot_resv(self):
        if not self.cola_network.is_connected:
            self.cola_say_new_line_ledit("Network", "Net work failed.")
        else:
            msg = self.cola_network.recv_msg()
            if msg == '':
                msg = "Receive error."
            self.cola_say_new_line_ledit("Receive", msg)
        
        
    def cola_slot_setroi(self):
        self.pbtn_roi.disconnect()  # disconnect
        version = 2
        if version == 1:
            self.cola_extra_color.run_roi()
        else:
            self.dlg.run(self.cola_camera)
        self.pbtn_roi.clicked.connect(self.cola_slot_setroi)  # connect
    

    def cola_slot_send(self):
        """
        brief: send message to stm32
        """
        if not self.cola_network.is_connected:
            self.cola_say_new_line_ledit("Send", "Net work failed.")
        else:
            msg = self.tedit_send_msg.toPlainText()
            self.cola_network.send_msg(msg, self.bbox_send_char.currentText())
            self.cola_say_new_line_ledit("Send", msg)


    def cola_slot_solve(self):
        """
        brief: Init lable and show
        """
        # self.label_tensor = self.cola_cluster.run(self.color_tensor)
        self.label_tensor = self.cola_cluster.run(self.color_tensor)
        self.check_label()
        self.cola_show_all_color_label(True)
        try:
            self.msg = ColaSolver.rubiks_solver(self.label_tensor)
            self.cola_say_new_line_ledit("Steps", len(self.msg))
            self.tedit_send_msg.setText(self.msg)
        except Exception as ex:
            print(ex)
            self.msg = "Solve error."
        code = ColaSolver.get_code(self.label_tensor)
        self.cola_say_new_line_ledit("Cube String", code)
        self.cola_say_new_line_ledit("Result", self.msg)
        

    def cola_slot_sample(self):
        """
        brief: sample the color of image, fill self.color_tensor -> update color show
        """
        self.color_tensor = self.cola_extra_color.extra_color()
        self.cola_show_all_color_label(False)
        self.cola_say_new_line_ledit("Sample", "Sample succeed.")
    
    
    def cola_show_all_color_label(self, is_show_label=False):
        """
        brief: show color to label_show with (color_tensor, label_tensor)
        color_all_face, labels
        """
        # draw the result   
        img = np.ones(shape=(500, 700, 3), dtype=np.ubyte) * 255
        delta = 48
        coords_face_start = np.asarray([1, 0, 2, 1, 1, 1, 1, 2, 0, 1, 3, 1], dtype=np.int32).reshape(6, 2) * 3.5 * delta  # 6 x 2
        globale_row_start = 20
        globale_col_start = 10

        def draw_a_face(img, color, label, col_start, row_start, is_show_label):
            for idx_row in range(3):
                for idx_col in range(3):
                    row = int(row_start + 0.5 * delta + idx_row * delta)
                    col = int(col_start + 0.5 * delta + idx_col * delta)
                    cv2.circle(img, (col, row), int(delta / 2), color[idx_row, idx_col].tolist(), -1)  # draw circle
                    if is_show_label:
                        cv2.putText(img, "{}".format(label[idx_row, idx_col]), (col - 5, row + 3), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)
        
        for idx_face in range(6):
            draw_a_face(
                img, 
                self.color_tensor[idx_face], 
                self.label_tensor[idx_face], 
                coords_face_start[idx_face][0] + globale_col_start, 
                coords_face_start[idx_face][1] + globale_row_start, 
                is_show_label
            )
        
        # check label
        error_dict = dict()
        for idx_color in range(6):
            num = int(np.sum(self.label_tensor == idx_color))
            if num != 9:
                error_dict[idx_color] = num
        if len(error_dict) != 0:
            info = "ERROR: {}".format(str(error_dict))
            cv2.putText(img, info, (10, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1)
        
        # set to image
        self.label_show.setPixmap((numpy_image_to_pixmap(img)))

        
    def initUI(self):
        # default char begin is NULL
        self.bbox_send_char.setCurrentIndex(2)
        
        # set to center
        screen = QDesktopWidget().screenGeometry()
        self.move(int((screen.width() - self.size_w ) / 2), int((screen.height() - self.size_h) / 2))
        
        # init sample image
        self.cola_slot_sample()
        
        # fixed window size
        self.setFixedSize(self.size_w, self.size_h)
        
        self.setWindowIcon(QIcon("./cola_store/cube.jpg"))
        t1 = Thread(target=self.cola_show_img)
        t1.start()
        
        # push button
        self.pbtn_reconnect.setFixedSize(120, 50)
        self.pbtn_roi.setFixedSize(120, 50)
        
        # push button 2 
        self.pbtn_sample.setFixedSize(100, 50)
        self.pbtn_store_sample.setFixedSize(100, 50)
        self.pbtn_solve.setFixedSize(100, 50)
        self.pbtn_send.setFixedSize(100, 50)
        self.pbtn_recv.setFixedSize(100, 50)
        
        # for lay out
        self.vlayout_rightlist.setAlignment(Qt.AlignCenter)
        self.vlayout_rightlist.setContentsMargins(30, 11, 11, 11)
        
        # show label
        self.label_show.setFixedSize(700, 500)
        background_color = QColor()
        background_color.setNamedColor('#FF00FF')
        pe = QPalette()
        pe.setColor(QPalette.WindowText, background_color)
        self.label_show.setPalette(pe)
        self.label_show.setAutoFillBackground(True)
        self.label_show.setAlignment(Qt.AlignCenter)
        
        # pbtn no border
        self.pbtn_running.setStyleSheet(
            '''QPushButton{background:#a8ffbc;border-radius:5px;}QPushButton:hover{background:#e6402e;}'''
        )
        self.pbtn_reconnect.setStyleSheet(
            '''QPushButton{background:#77ac92;border-radius:5px;}QPushButton:hover{background:#125957;}'''
        )
        self.pbtn_roi.setStyleSheet(
            '''QPushButton{background:#77ac92;border-radius:5px;}QPushButton:hover{background:#125957;}'''
        )
        self.pbtn_sample.setStyleSheet(
            '''QPushButton{background:#77ac92;border-radius:5px;}QPushButton:hover{background:#125957;}'''
        )
        self.pbtn_store_sample.setStyleSheet(
            '''QPushButton{background:#77ac92;border-radius:5px;}QPushButton:hover{background:#125957;}'''
        )
        self.pbtn_solve.setStyleSheet(
            '''QPushButton{background:#77ac92;border-radius:5px;}QPushButton:hover{background:#125957;}'''
        )
        self.pbtn_send.setStyleSheet(
            '''QPushButton{background:#77ac92;border-radius:5px;}QPushButton:hover{background:#125957;}'''
        )
        self.pbtn_recv.setStyleSheet(
            '''QPushButton{background:#77ac92;border-radius:5px;}QPushButton:hover{background:#125957;}'''
        )
        
    
    def cola_say_new_line_ledit(self, host, msg):
        old_data = self.tedit_show.toPlainText().split('\n')
        if len(old_data) >= 8:
            self.tedit_show.setText("")    
        self.tedit_show.append("{} >>\n{}".format(host, msg))
    
        
    def cola_put_new_line_ledit(self, msg):
        self.tedit_show.append(msg)
    
    
    def cola_show_net_state(self):
        style_red = "border-radius: 8px; border:0px solid black;background:red"
        style_green = "border-radius: 8px;  border:0px solid black;background:green"
        is_connect = self.cola_network.state_is_connect()
        style = style_green if is_connect else style_red
        self.label_net.setStyleSheet(style)
        self.label_net.setText("")
        self.label_net.setFixedSize(120, 40)
        self.label_net.setAlignment(Qt.AlignCenter)
        
    
    def closeEvent(self, envent):
        self.flag_show_img = False
    
    
    def cola_show_img(self):
        def set_image(img, label_img):
            img_h, img_w, img_c = img.shape
            num_byte_per_line = img_c * img_w
            q_img  = QImage(img.data, img_w, img_h, num_byte_per_line, QImage.Format_BGR888 )
            pix = QPixmap.fromImage(q_img)
            label_img.setPixmap(pix)
            
        
        label_list = [self.label_img0, self.label_img1, self.label_img2, self.label_img3]
        while self.flag_show_img:
            for i in range(4):
                set_image(self.cola_extra_color.img_with_roi(i), label_list[i])
            cv2.waitKey(200)
        
    def __del__(self):
        np.save("./cola_store/color_tensor", self.color_tensor)
                

def get_conf():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", '--target', type=str, choices=['run', 'show'], default='run')
    parser.add_argument('-is-test', action='store_true')
    return parser.parse_args()

            
def main():
    conf = get_conf()
    if conf.target == 'run':
        app = QApplication(sys.argv)
        cola_win = ColaMainWindow(is_test=conf.is_test)
        cola_win.show()
        sys.exit(app.exec_())
    elif conf.target == 'show':
        show_all_cammera()


def reset_cammera_configuration():
    cola_camera = ColaCamera()


if __name__ == "__main__":
    main()
    