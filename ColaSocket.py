import socket
import threading
import numpy as np

socket.setdefaulttimeout(0.5)
        
class ColaSocket():
    def __init__(self, IP: str="127.0.0.1", port=1233):
        self.IP = IP
        self.port = port
        self.s = None
        self.is_connected = False
    
    
    def update_net_address(self, ip, port):
        # self.disconnect()
        self.port = int(port)
        self.IP = ip
        # self.connect()
    
    
    def state_is_connect(self):
        """
        retrun: connect:True, disconnect:False
        """
        return self.is_connected


    def spell_and_wait(self):
        """
        send a comfirm message to server and listen
        """
        if self.s is None:
            print("Socket is None!")
            self.is_connected = False
            return
        confirm_msg = "COLA_CHENG"
        try:
            self.send_msg(confirm_msg)
            data = self.recv_msg()
            if data == "":
                self.is_connected = False
        except Exception as ex:
            print("Spell_and_wait:{}".format(ex))
            self.is_connected = False


    def reconnect(self, IP, port):
        """
        brief: compare ip and port, and check
        """
        socket.setdefaulttimeout(4)
        if self.IP != IP or self.port != port:  # new port
            print(self.IP, IP, self.port, port)
        self.disconnect()
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.IP = IP
        self.port = port
        try:
            self.s.connect((self.IP, self.port))
            self.is_connected = True
        except Exception as ex:
            print(ex)
            self.is_connected = False
            socket.setdefaulttimeout(0.5)
        else:
            print("Reconnected.")
        socket.setdefaulttimeout(0.5)
        
    
    def try_connect(self):
        try:
            self.s.connect((self.IP, self.port))
        except Exception as ex:
            print("Try_connect: {}".format(ex))
            self.is_connected = False
        else:
            print("Reconnected: {}/{}".format(self.IP, self.port))
            self.is_connected = True


    def connect(self):
        if self.s is not None:
            self.spell_and_wait()
            if self.is_connected:
                return
            self.disconnect()
        print(type(self.port))
        # reconnect
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Reconnect")
        try:
            self.s.connect((self.IP, self.port))
        except Exception as ex:
            self.is_connected = False
            print("Connect: {}".format(ex))
        else:
            self.is_connected = True
            print("ReconnectEnd")
        
    
    def disconnect(self):
        if self.s is None:
            self.is_connected = False
            return
        try:
            self.s.close()
            self.s = None
            self.is_connected = False
        except Exception as ex:
            print(ex)
        else:
            print("Disconnect: finished!")
        self.s = None
    
    def send_msg(self, msg: str, begin_char=''):
        if begin_char != "NULL":
            msg = begin_char + msg
        
        if self.s is None:
            print("Please connect net work.")
            return
        try:
            msg = msg.encode()
            # print(msg)
            self.s.sendall(msg)
        except Exception as ex:
            print("Send_msg:{}".format(ex))
        else:
            print("Send finished")

    def recv_msg(self):
        """
        retrun: str of message (ascii)
        """
        try:
            data = self.s.recv(1024)        
        except Exception as ex:
            pass
            # print("Recv_msg: {}".format(ex))
        else:
            msg = data.decode()
            print("Received: {}".format(msg))
            return msg
        return ""

if __name__ == "__main__":
    cola_socket = ColaSocket("192.168.1.101", 1234)
    cola_socket.connect()
    cola_socket.send_msg("8196280953704805970825971354859173740305485292364168093064153541815951537418195")
    
    
        

