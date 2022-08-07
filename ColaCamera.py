import cv2
import numpy as np
from threading import Thread
import os
import sys
import yaml
# Cola import
from ColaLib import cola_show_image
from ColaLib import mouse_callback
from ColaCapState import ColaCapState


class ColaCamera:
    def __init__(self, is_test=True):
        self.cap = [0] * 4  # four capture object D U FR BL
        self.cap_map_name = {
            'D': 0, 
            'U': 1, 
            'FR': 2, 
            'RF': 2, 
            'BL': 3, 
            'LB': 3
        }
        # self.cap_map_name = {
        #     "B": 0, 
        #     "F": 1, 
        #     "RD": 2, 
        #     "DR": 2, 
        #     "UL": 3, 
        #     "LU": 3
        # }
        self.cap_state = [ColaCapState(), ColaCapState(), ColaCapState(), ColaCapState()]  # [0]: B
        # for i in range(4):
        #     self.cap_state[i].init(i)
        # print(self.cap_state[0].rotation)
        self.faces_id = [1, 2, 2, 0, 3, 3]  # faces_id[idx_face] -> id of camera -> capture -> image
        self.faces_roi = np.zeros(shape=(6, 3, 3, 2), dtype=np.int32)  # [idx_face][row][col] -> roi center
        self.is_test = is_test
        
        if self.is_test:
            sys.stdin = open("data_test.in", "r")
            self.init_test()
        else:
            self.load_configuration("./cola_store/config_camera.yaml")  # replace init function


    def load_configuration(self, filename):
        with open(filename, mode='r') as fin:
            data = yaml.safe_load(fin)
        print(data)
        num_confirmed = 0
        id = int(0)
        idx_seq = int(0)
        seq = data['seq']
        while num_confirmed < 4:
            cap = cv2.VideoCapture(id)
            ret, frame = cap.read()
            if not ret:
                id += 1
                continue
            # Not our eye
            name_face = seq[idx_seq]
            idx_seq += 1
            if name_face == 'NO':
                cap.release()
                id += 1
                continue
            idx_cap = self.cap_map_name[name_face]
            self.cap[idx_cap] = cap
            tmp = data[name_face]
            print(tmp)
            coord_start = tmp[0:2]
            coord_end = tmp[2:4]
            rotation = tmp[4]
            self.cap_state[idx_cap].init2(
                coord_start[0], coord_start[1], \
                min(abs(coord_start[0] - coord_end[0]), abs(coord_start[1] - coord_end[1])), \
                rotation, \
                idx_cap)
        
            id += 1    
            num_confirmed += 1
        
    
    def get_cap_frame(self, cap_name_id):
        """
        """
        if self.is_test:
            filename_list = [
                "./cola_store/snap2_image_B.jpg", 
                "./cola_store/snap2_image_F.jpg", 
                "./cola_store/snap2_image_DR.jpg", 
                "./cola_store/snap2_image_UL.jpg", 
            ]
            filename_list = [
                "./cola_store/snap2_image_BL.jpg", 
                "./cola_store/snap2_image_D.jpg", 
                "./cola_store/snap2_image_FR.jpg", 
                "./cola_store/snap2_image_U.jpg", 
            ]  
            # filename_list = [
            #     "./cola_store/snap3_image_B.jpg", 
            #     "./cola_store/snap3_image_F.jpg", 
            #     "./cola_store/snap3_image_DR.jpg", 
            #     "./cola_store/snap3_image_UL.jpg", 
            # ]            
            img = cv2.imread(filename_list[cap_name_id])
            return img
        else:
            """
            cap_name_id: idx of cap or name of cap
            """
            idx_cap = cap_name_id
            if isinstance(cap_name_id, str):
                idx_cap = self.cap_map_name[idx_cap]
            ret, frame = self.cap[idx_cap].read()
            if not ret:
                return np.zeros(shape=(600, 600, 3),dtype=np.ubyte)                
            return frame
        
    
    def get_cap_frame_with_rotation(self, idx_cap):
        img = self.get_cap_frame(idx_cap)
        mesh_w, mesh_h = self.cap_state[idx_cap].get_image_range(img)
        img = img[mesh_h, mesh_w]
        img = self.image_rotation(img, self.cap_state[idx_cap].rotation)
        img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_LANCZOS4)
        return img
    
    
    def image_rotation(self, img, angle):
        """
        roatate image with 90 180 270
        """
        if angle == 90:
            return np.flip(np.swapaxes(img, 0, 1), 0)
        elif angle == 180:
            return np.flip(np.flip(img, 0), 1)
        elif angle == 270:
            return np.flip(np.swapaxes(img, 0, 1), 1)
        else:
            return img
    
    
    def get_face_image(self, idx_face):
        """
        retrun the squre image of idx_face
        """
        idx_cap = self.faces_id[idx_face]
        img = self.get_cap_frame(idx_cap)
        mesh_w, mesh_h = self.cap_state[idx_cap].get_image_range(img)
        img = img[mesh_h, mesh_w]
        img = self.image_rotation(img, self.cap_state[idx_cap].rotation)
        img = cv2.resize(img, (600, 600), interpolation=cv2.INTER_LANCZOS4)
        return img


    def set_roi(self):
        """
        set roi -> fill self.faces_roi
        """
        pass
    
    
    def init_test(self):
        for i in range(4):
            flag = input("No[0] Yes[1]:")
            name_face = input("Input name of face:")
            idx_cap = self.cap_map_name[name_face]
            coord_start = [int(item) for item in input("Coordinates of start:").split()]
            coord_end = [int(item) for item in input("Coordinates of end:").split()]
            rotation = int(input("Rotation(90/180/270):"))
            self.cap_state[idx_cap].init2(coord_start[0], coord_start[1], \
                min(abs(coord_start[0] - coord_end[0]), abs(coord_start[1] - coord_end[1])), \
                    rotation, \
                    idx_cap)
            
    
    def init(self):
        """
        initial cap/cap_state/faces_id/faces_roi
        """
        num_confirmed = 0
        id = int(0)
        while num_confirmed < 4:
            cap = cv2.VideoCapture(id)
            ret, frame = cap.read()
            if not ret:
                num_confirmed += 1
                continue
            
            # show image and require input info
            # t = Thread(target=cola_show_image, args=(None, "WIN", "", cap))  # show image 
            # t.start()
            flag = input("No[0] Yes[1]:")
            if flag == '0':
                # t.join()
                id += 1
                cap.release()
                continue
                
            name_face = input("Input name of face:")
            idx_cap = self.cap_map_name[name_face]
            self.cap[idx_cap] = cap
            coord_start = [int(item) for item in input("Coordinates of start:").split()]
            coord_end = [int(item) for item in input("Coordinates of end:").split()]
            rotation = int(input("Rotation(90/180/270):"))
            self.cap_state[idx_cap].init2(coord_start[0], coord_start[1], \
                min(abs(coord_start[0] - coord_end[0]), abs(coord_start[1] - coord_end[1])), \
                    rotation, \
                    idx_cap)
            
            id += 1
            num_confirmed += 1
            # t.join()
            

def show_all_cammera():
    """
    close all capture, and show all
    """
    id = int(0)
    is_not_return = True
    cv2.namedWindow("WIN")
    coord = [0, 1]
    cv2.setMouseCallback("WIN", mouse_callback, coord)
    while is_not_return:
        cap = cv2.VideoCapture(id)
        id += 1
        ret, img = cap.read()
        if not ret:
            cap.release()
            continue
        print(id)
        while True:
            ret, img = cap.read()
            frame = img.copy()
            info = "{}:({},{})".format(id, coord[0], coord[1])
            cv2.putText(frame, info, (20, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0), 1)
            cv2.imshow("WIN", frame)
            key = cv2.waitKey(30)
            if key == ord('q'):
                is_not_return = False
                break
            elif key == ord('n'):
                break
            
            
"""

"""            
                        

if __name__ == "__main__":
    c_camera = ColaCamera(is_test=False)
    for idx_face in range(6):
        while True:
            img = c_camera.get_face_image(idx_face)
            cv2.imshow("WIN", img)
            key = cv2.waitKey(30)
            if key == ord('q'):
                break
    cv2.destroyWindow("WIN")
        
    
    
            
        