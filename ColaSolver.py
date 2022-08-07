import kociemba
import os
import numpy as np


time_duration_ms = 0
def get_time_duration():
    global time_duration_ms
    tmp = time_duration_ms
    time_duration_ms = 0
    return tmp


def trans_sequence(seq1):
    """
    brief: six hands to tow hands
    """
    cmd = "TransSequence.exe \"{}\"".format(seq1)
    path_cwd = os.getcwd()
    os.chdir('./src_trans/')
    fp = os.popen(cmd, "r")
    os.chdir(path_cwd)
    return fp.read().split( )



def calculate_seq_time(seq):
    """
    brief: prediction the time to do the cube
    seq: list
    """
    base_time_vec = np.asarray([200, 100, 230, 270], dtype=np.int32) 
    times_vec = np.zeros(shape=(4, 1), dtype=np.int32)
    for item in seq:
        # print(item[2])
        if item[3] == '2':
            times_vec[1] += 1
        elif item[3] == '1' or item[3] == '3':
            times_vec[0] += 1
        elif item[3] == 'C':
            times_vec[2] += 1
        elif item[3] == 'O':
             times_vec[3] += 1
    return np.dot(base_time_vec, times_vec)


def get_code(label_faces):
    """
    brief: label number coded to str for tow steps algorithm
    """
    label_faces = label_faces.reshape(6, 3, 3)
    str_vec = ['U', 'R', 'F', 'D', 'L', 'B']
    code = [ "{}".format(str_vec[label_faces[idx_face, row, col]]) \
        for idx_face in range(6) \
            for row in range(3) \
                for col in range(3) ]
    code = ''.join(code)
    return code


def rubiks_solver(label_faces):
    """
    label_faces: ndarray, shape=(6, 3, 3), range [0, 6)
    """
    # encoding
    code = get_code(label_faces)
    print(code)
    ans1_seq = kociemba.solve(code)
    ans1_seq = ans1_seq.replace('\'', '3')
    ans1_seq = ans1_seq.split(' ')
    ans1_seq = [  "{}1".format(item) if len(item) == 1 else item for item in ans1_seq]
    ans1_seq = ' '.join(ans1_seq)
    ans1_seq = ans1_seq + ' '
    # print(ans1_seq, end="---")
    ans2_seq = trans_sequence(ans1_seq)  # ML_1 ...
    
    # translate to asii 0~a
    map_dict = {
        "M_L1": '0',
        "M_L2": '1',
        "M_L3": '2',
        "M_LO": '3', 
        "M_LC": '4', 
        "M_R1": '5',
        "M_R2": '6',
        "M_R3": '7',
        "M_RO": '8',
        "M_RC": '9'
    }
    ans3_seq = [ map_dict[key] for key in ans2_seq ]
    msg = ''.join(ans3_seq)
    
    time_total = calculate_seq_time(ans2_seq)
    time_duration_ms = time_total
    # print("Total time: {:.3}s".format(float(time_total) / 1000.0))

    return msg


if __name__ == "__main__":
    pass
