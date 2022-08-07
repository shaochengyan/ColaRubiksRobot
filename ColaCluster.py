import numpy as np
import cv2
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import NuSVC, LinearSVC




class ColaCluster:
    def __init__(self, algorithm="KNN"):
        self.algorithm = algorithm
        self.train_X, self.train_y = self.load_data()
        self.ro_svm = self._get_nonlinear_svm_ro()
        self.neigh = KNeighborsClassifier(n_neighbors=1, weights='distance')
        self.neigh.fit(self.train_X, self.train_y)
        self.neigh_ro = self.get_knn_model()
        
        # version 2
        self.svm_for_ro_other = self.get_linear_svc_for_ro_other()
        
    
    def reload_knn_ro_model(self):
        print("ColaCluster: Reload knn model")
        self.neigh_ro = self.get_knn_model()
        
        
    def _label_rgb_kmeans(self, color_tensor):
        """
        color_tensor: shape=(6, 3, 3, 3)
        # todo: init the centroid 
        """
        k = 6
        estimator = KMeans(n_clusters=k, max_iter=4000, init='k-means++', n_init=50)
        estimator.fit(color_tensor)
        
        # the idx of face set to label, face 0 -> label 0
        label1 = estimator.labels_.reshape(6, 3, 3)
        label2 = np.ones_like(label1) * -1
        for idx_face in range(6):
            label_tmp = label1[idx_face, 1, 1]
            label2[label1 == label_tmp] = idx_face
            
        return label2
    
    
    def load_data(self):
        color_name_vec = ["Yellow", "Orange", "Green", "White", "Red", "Blue"]
        train_X = None
        train_y = None
        for idx_face in range(6):
            filename = "./cola_store/color_mat_{}.txt".format(color_name_vec[idx_face])
            if not os.path.isfile(filename):
                continue
            else:
                X_tmp = np.loadtxt(filename)
                if len(X_tmp) == 0:
                    continue
            y_tmp = np.ones(shape=(X_tmp.shape[0], 1), dtype=np.int32) * idx_face
            if train_X is None:
                train_X = X_tmp
                train_y = y_tmp
            else:
                train_X = np.row_stack((train_X, X_tmp))
                train_y = np.row_stack((train_y, y_tmp))
        return train_X, train_y

    
    
    def _label_rgb_knn(self, color_tensor):
        """
        color_tensor: shape=(N, 3) ubyte
        """
        # print(neigh.predict(self.train_X))
        y_hat1 = self.neigh.predict(color_tensor)
        
        # using linear svm to classify orange and red in RGB
        bidx_1 = np.logical_or(y_hat1 == 1, y_hat1 == 4)
        if np.any(bidx_1):
            ro_X = color_tensor[bidx_1]
            # print(ro_X)
            y_hat2 = self.ro_svm.predict(ro_X)
            y_hat1[bidx_1] = y_hat2
        return y_hat1
        
        
    def _get_nonlinear_svm_ro(self):
        """
        retrun: a Linear Support Vector Classification for red and orange color in RGB space
        """
        # Load data 
        color_name_vec = ["Orange", "Red"]
        filename_base = "./cola_store/color_mat_{}.txt"
        filename_orange = filename_base.format(color_name_vec[0])
        filename_red = filename_base.format(color_name_vec[1])
        
        train_X = np.asarray([1, 1, 1], dtype=np.ubyte)
        tmp_X = np.asarray([1, 1, 1], dtype=np.ubyte)
        if os.path.isfile(filename_orange):
            train_X = np.loadtxt(filename_orange)
            train_y = np.ones(shape=(train_X.shape[0], 1), dtype=np.int32) * 1

        if os.path.isfile(filename_red):
            tmp_X = np.loadtxt(filename_red)
            tmp_y = np.ones(shape=(tmp_X.shape[0], 1), dtype=np.int32) * 4
        
        
        train_X = np.row_stack((train_X, tmp_X))
        train_y = np.row_stack((train_y, tmp_y))
        
        # create svm
        # svc = LinearSVC()
        svc = NuSVC()
        svc.fit(train_X, train_y)
        # print(svc.predict(train_X))
        return svc
        
        
    def run(self, color_tensor):
        """
        color_tensor: store pixle of BGR, ndarray, shape=(6, 3, 3, 3)
        retrun cluster_label: ndarray, shape=(6, 3, 3)
        """
        assert color_tensor.shape == (6, 3, 3, 3)
        # np.savetxt("tmp_rgb.txt", color_tensor[-1::-1].reshape(-1,3))
        
        # todo:algorithm to lable the pxile
        # test1: lable by bgr using k-menas
        
        # version 1
        # cluster_label = self._label_rgb_kmeans(color_tensor.reshape(-1, 3)).reshape(6, 3, 3)
        # if not self.check_label_tensor_right(cluster_label):
        #     print("Cluster error.")
        #     cluster_label = self._label_rgb_knn(color_tensor.reshape(-1, 3)).reshape(6, 3, 3)  # not so good
        #     if not self.check_label_tensor_right(cluster_label):
        #         print("KNN error.")
        #     else:
        #         print("KNN right.")
        # else:
        #     print("Cluster right.")
        
        # version 2: using cuboid and knn for ro
        color_tensor = color_tensor.reshape(-1, 3)
        cluster_label = np.zeros(shape=(6, 3, 3), dtype=np.int32).reshape(-1)
        for i in range(color_tensor.shape[0]):
            cluster_label[i] = self.classify_hsv(color_tensor[i])
        
        
        label_1 = cluster_label.reshape(6, 3, 3)
        label_2 = label_1.copy()
        for idx_face in range(6):
            center_label = label_1[idx_face, 1, 1]
            label_2[label_1 == center_label] = idx_face 
            
        return label_2
    
    
    def classify_rgb(self, color_tensor):
        """
        color_tensor = (N, 3)
        classify rgb color by knn and ro_svm
        """
        return self._label_rgb_knn(color_tensor).reshape(-1)
        
        
    
    
    def check_label_tensor_right(self, label_tensor):
        """
        label_tensor: shape=(6, 3, 3)
        retrun: true if label is right, otherwise false
        """
        for label in range(6):
            if np.sum(label_tensor == label) != 9:
                return False
        return True
    
    
    def is_in_cuboid(self, cube_para, color_hsv, rate=0.01):
        """
        cube_par: ndarray shape=(2, 3), [0]->min(hsv) [1] -> max(hsv)
        color_hsv: ndarray hsv color
        retrun: True if this color_hsv in cube
        """
        color_hsv = np.asarray(color_hsv)
        border = np.abs(cube_para[1] - cube_para[0]) * rate
        # print(border)
        return np.all(np.logical_and(cube_para[0] <= color_hsv + border, cube_para[1] + border >= color_hsv)) 
    
    def get_data_for_ro(self):
        color_name_vec = ["Orange", "Red"]
        filename_base = "./cola_store/color_mat_{}.txt"
        train_X = np.loadtxt(filename_base.format(color_name_vec[0]))
        train_y = np.ones(shape=(train_X.shape[0], 1), dtype=np.int32) * 1
        tmp_X = np.loadtxt(filename_base.format(color_name_vec[1]))
        tmp_y = np.ones(shape=(tmp_X.shape[0], 1), dtype=np.int32) * 4
        train_X = np.row_stack((train_X, tmp_X))
        train_y = np.row_stack((train_y, tmp_y))
        return train_X, train_y.reshape(-1, )
    
    
    def get_knn_model(self):
        # get dataset
        train_X, train_y = self.get_data_for_ro()
        
        # train_X = cv2.cvtColor(train_X.reshape(1, -1, 3).astype(np.ubyte), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        neigh_ro = KNeighborsClassifier(n_neighbors=1, weights='distance')
        neigh_ro.fit(train_X, train_y)
        
        return neigh_ro
    
    def get_all_data_bgr_for_ro_and_other(self):
        color_name_vec = ["Yellow", "Orange", "Green", "White", "Red", "Blue"]
        X, y = None, None
        for idx_color in range(len(color_name_vec)):
            tmp_X = np.loadtxt("./cola_store/color_mat_{}.txt".format(color_name_vec[idx_color]))
            tmp_y = np.zeros(shape=(tmp_X.shape[0], 1), dtype=np.int32) + idx_color
            if X is None:
                X, y = tmp_X, tmp_y
            else:
                X = np.vstack((X, tmp_X))
                y = np.vstack((y, tmp_y))
        y = y.reshape(-1)
        idx_ro = np.logical_or(y == 1, y == 4)
        y[:] = 0
        y[idx_ro] = 1
        return X, y
    

    def get_linear_svc_for_ro_other(self):
        """
        X: shape=(N, 3)
        y: 0->other, 1->red or orange
        """
        X, y = self.get_all_data_bgr_for_ro_and_other()
        svc_model = LinearSVC(max_iter=10000)
        svc_model.fit(X, y)
        return svc_model
    

    def classify_hsv(self, color_tensor):
        """
        color_tensor: bgr (3, ) ndarray 
        """
        # 173   5 198
        # white -> red 27  56 123
        # white -> red 28  76 111
        cube_para = {
            "Yellow": np.asarray([
                [30, 16, 240], 
                [50, 71, 253], 
                [30, 71, 0], 
                [50, 255, 255], 
            ]), 
            "Green": np.asarray([
                [51, 65, 0], 
                [80, 255, 255]
            ]), 
            "Blue": np.asarray([
                [81, 65, 0], 
                [120, 255, 255]
            ]), 
            "White": np.asarray([
                [0, 0, 0], 
                [30, 80, 255], 
                [33, 0, 0], 
                [48, 71, 145], 
                [31, 0, 0], 
                [48, 55, 255], 
                [49, 0, 0], 
                [160, 50, 255], 
                
            ])
        }
        color_name_vec = ["Yellow", "Green", "White", "Blue"]
        i_to_idx_colo_vec = [0, 2, 3, 5]
        color_hsv = cv2.cvtColor(
            color_tensor.reshape(1, -1, 3), 
            cv2.COLOR_BGR2HSV).reshape(3)
        for i in range(4):
            para_mat = cube_para[color_name_vec[i]]
            for idx_cuboid in range(0, para_mat.shape[0], 2):
                para = para_mat[idx_cuboid:(idx_cuboid+2)]
                # print(para)
                # print(para.reshape(2, 3))
                if self.is_in_cuboid(
                    para, 
                    color_hsv):
                    return i_to_idx_colo_vec[i]
            
        # use knn to label red and orange
        label = self.neigh_ro.predict(color_tensor.reshape(1, 3))
        
        return int(label)


    def run2(self, color_tensor):
        """
        Steps:
            1. 
        """
        pass
    

def run_classification():
    from ColaLib import mouse_callback
    
    color_name_vec = ["Yellow", "Orange", "Green", "White", "Red", "Blue"]
    coord = [0, 0]
    cv2.namedWindow("WIN")
    cv2.setMouseCallback("WIN", mouse_callback, coord)
    cap = cv2.VideoCapture(1)
    cola_cluster = ColaCluster()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # show class
        color_tensor = frame[coord[1], coord[0]]
        label = cola_cluster.classify_hsv(color_tensor)
        info = "{}".format(color_name_vec[label])
        cv2.putText(frame, info, (20, 20), 
            cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0), 1)
        cv2.imshow("WIN", frame)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

        
def test1():
    cola_cluster = ColaCluster()
    color_tensor = np.random.randint(0, 256, size=(54, 3), dtype=np.ubyte)
    print(color_tensor)
    result = cola_cluster.run(color_tensor.reshape(6, 3, 3, 3))
    print(result)


def test2():
    run_classification()

if __name__ == '__main__':
    # test1()
    test2()