import cv2
import numpy as np
from threading import Thread
import os
import sys
# Qt import
from PyQt5.QtGui import QImage, QPixmap


def numpy_image_to_pixmap(img):
    img_h, img_w, img_c = img.shape
    num_byte_per_line = img_c * img_w
    q_img  = QImage(img.data, img_w, img_h, num_byte_per_line, QImage.Format_BGR888 )
    return QPixmap.fromImage(q_img)

# click the windows recall
def mouse_callback(event, w, h, flags, coord):
    if event == cv2.EVENT_LBUTTONDOWN:
        coord[0] = w
        coord[1] = h

def cola_show_image(img=None, name_win="WIN", info="", cap=None):
    """
    Show an image or capture
    """
    coord = [0, 0]
    cv2.namedWindow(name_win)
    cv2.setMouseCallback(name_win, mouse_callback, coord)
    while True:
        if cap is not None:
            ret, img = cap.read()
        info_tmp = "{} wh:({}, {})".format(info, coord[0], coord[1])
        cv2.putText(img, info_tmp, (20, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, .7, (100, 100, 255), 1)
        cv2.imshow(name_win, img)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break
    cv2.destroyWindow(name_win)
        
