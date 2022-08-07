import cv2
import numpy as np
import os
# Cola import
from ColaLib import mouse_callback


class ColaExtraColor:
    def __init__(self, cola_camera):
        self.filename = "./cola_store/coords_all_faces.npy"
        self.coord_faces = None
        self.char_faces = list('URFDLB')
        self.cola_camera = cola_camera
        if os.path.isfile(self.filename):
            self.coord_faces = np.load(self.filename)
        else:
            self.run_roi()
            
    
    def __del__(self):
        np.save(self.filename, self.coord_faces)
    
    
    def save_parameters(self):
        np.save(self.filename, self.coord_faces)
    
    
    def _draw_roi(self, img, idx_face):
        """
        idx_face -> coord: 3x3x2 for an image
        """
        radiux = 5
        for row in range(3):
            for col in range(3):
                coord = self.coord_faces[idx_face, row, col]   # w, h
                cv2.circle(img, coord, radiux, [200, 0, 0], -1)
                cv2.putText(img, str(1 + row * 3 + col), coord + 10, cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 1)
    
    
    def get_idx_face_img_with_roi(self, idx_face):
        img = self.cola_camera.get_face_image(idx_face)
        self._draw_roi(img, coord)
        return img
    
        
    def img_with_roi(self, idx_cap):
        """
        draw the roi are to img
        """
        radiux = 5
        img = self.cola_camera.get_cap_frame_with_rotation(idx_cap)
        idx_cap_to_idx_face_array = np.asarray(self.cola_camera.faces_id)
        for idx_face in np.argwhere(idx_cap_to_idx_face_array == idx_cap).reshape(-1):
            self._draw_roi(img, idx_face)
        return img
        
    
        
    def extra_color(self):
        """
        extrac color of image of idx_face
        """        
        color_faces = np.zeros(shape=(6, 3, 3, 3), dtype=np.ubyte)
        radiux = 5
        for idx_face in range(6):
            img = self.cola_camera.get_face_image(idx_face)
            for row in range(3):
                for col in range(3):
                    coord = self.coord_faces[idx_face, row, col]   # w, h
                    # get the mean pixel vector of the circle with center of coord
                    coord_w, coord_h = np.meshgrid(np.arange(coord[0] - radiux, coord[0] + radiux), 
                                                   np.arange(coord[1] - radiux, coord[1] + radiux))
                    pixel_circle = img[coord_h, coord_w]
                    color_faces[idx_face, row, col] = np.mean(pixel_circle, axis=(0, 1))
        return color_faces
        
        
        
def trans_bgr_to_hsv(color_list=None):
    if color_list is None:
        color_list = ["Yellow", "Orange", "Green", "White", "Red", "Blue"]
    filebase = "./cola_store/color_mat_{}.txt"
    bgr_vec = []
    hsv_vec = []
    for c in color_list:
        filename = filebase.format(c)
        filename_hsv = filebase.format(c + "_hsv")
        data = np.loadtxt(filename).astype(np.ubyte)
        bgr_vec.append(data)
        data_hsv = cv2.cvtColor(data.reshape(-1, len(data), 3), cv2.COLOR_BGR2HSV)
        hsv_vec.append(data_hsv.reshape(-1, 3))
        # save
        data_hsv = data_hsv.reshape(-1, 3)
        np.savetxt(filename_hsv, data_hsv, fmt="%2d")        

        
def sample_color(i_vec):
    """
    sample the same color of cube and store the color vector
    """
    cap = cv2.VideoCapture(2)
    color_name_vec = ["Yellow", "Orange", "Green", "White", "Red", "Blue"]
    for idx_face in i_vec:
        filename = "./cola_store/color_mat_{}.txt".format(color_name_vec[idx_face])
        print(color_name_vec[idx_face])
        print(filename)
        color_tmp = np.zeros(shape=(3, ), dtype=np.ubyte)
        color_vec = None
        if os.path.isfile(filename):
            color_vec = np.loadtxt(filename)
        coord = [240, 230]
        name_win = "WIN"
        cv2.namedWindow(name_win)
        cv2.setMouseCallback(name_win, mouse_callback, coord)
        radiux = 5
        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            img_original = frame.copy()
            if ret == False:
                return
            cv2.circle(frame, coord, radiux, (0, 100, 0))
            cv2.circle(frame, (20, 20), radiux, color_tmp.tolist(), -1)
            cv2.imshow(name_win, frame)
            key = cv2.waitKey(30)
            if key == 13: # enter -> sample color
                coord_w, coord_h = np.meshgrid(np.arange(img_original.shape[1]), np.arange(img_original.shape[0]))
                bidx_img = (np.abs(coord_w - coord[0]) + np.abs(coord_h - coord[1])) < radiux
                pixel_circle = img_original[bidx_img]
                color_tmp = np.mean(pixel_circle, axis=0)
            elif key == ord('4'):  # save
                # else save the color vec           
                print(color_tmp)
                if color_vec is None:
                    color_vec = color_tmp
                else:
                    # print(color_vec)
                    color_vec = np.row_stack((color_vec, color_tmp))
            elif key == ord('q'):
                break

        filename = "./cola_store/color_mat_{}.txt".format(color_name_vec[idx_face])
        cv2.destroyWindow(name_win)
        color_vec = color_vec.astype(np.ubyte)
        np.savetxt(filename, color_vec)
        
    cap.release()
    trans_bgr_to_hsv()
        
        
if __name__ == "__main__":
    # i_vec = [0, 1, 2, 3, 4, 5]  # "Yellow", "Orange", "Green", "White", "Red", "Blue"
    i_vec = [0, 3]
    sample_color(i_vec)        
    