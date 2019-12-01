#!/usr/bin/python3

import _thread
import time
import threading
import socket
import cv2
import numpy as np
from PIL import Image
import sys
import codecs
import struct

import os
import time
import darknet


class YoloNet:
    netMain = None
    metaMain = None
    altNames = None
    configPath = "./cfg/yolov3.cfg"
    weightPath = "./yolo-weights/yolov3.weights"
    metaPath = "./cfg/coco.data"

    def __init__(self):
        pass

    def load_data(self):
        if not os.path.exists(YoloNet.configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(YoloNet.configPath) + "`")
        if not os.path.exists(YoloNet.weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(YoloNet.weightPath) + "`")
        if not os.path.exists(YoloNet.metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(YoloNet.metaPath) + "`")
        if YoloNet.netMain is None:
            YoloNet.netMain = darknet.load_net_custom(YoloNet.configPath.encode(
                "ascii"), YoloNet.weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if YoloNet.metaMain is None:
            YoloNet.metaMain = darknet.load_meta(YoloNet.metaPath.encode("ascii"))
        if YoloNet.altNames is None:
            try:
                with open(YoloNet.metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                      re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

    def convertBack(self, x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def cvDrawBoxes(self, detections, img):
        total_info = []
        for detection in detections:
            x, y, w, h = detection[2][0], \
                         detection[2][1], \
                         detection[2][2], \
                         detection[2][3]
            xmin, ymin, xmax, ymax = self.convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(img,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
            total_info.append((detection[0].decode(), detection[1] * 100, [xmin, ymin, xmax, ymax]))
        return img, total_info

    def detect(self, image):
        # Create an image we reuse for each detect
        darknet_image = darknet.make_image(darknet.network_width(YoloNet.netMain),
                                           darknet.network_height(YoloNet.netMain), 3)
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(self.netMain),  # 416
                                    darknet.network_height(self.netMain)),  # 416
                                   interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(self.netMain, self.metaMain, darknet_image, thresh=0.80)  # thresh=0.50识别的阈值
        # detections=  (b'bird', 0.9903159141540527, (172.501708984375, 188.70071411132812, 56.11371994018555, 177.9843292236328))
        # nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)

        boxed_image, total_info = self.cvDrawBoxes(detections, frame_resized)  # total_info [name ,pro,[左上，右下]]

        # print('total_info=', total_info)
        boxed_image = cv2.cvtColor(boxed_image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Demo', boxed_image)
        # cv2.waitKey(1)
        return total_info, boxed_image

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
print("socket start....")
receive_picture = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 第一步：初始化socket
send_msg = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 第一步：初始化socket

receive_picture.bind(('127.0.0.1', 9000))  # 第二步：绑定端口
send_msg.bind(('127.0.0.1', 9001))  # 第二步：绑定端口

receive_picture.listen(5)  # 第三步：监听端口
send_msg.listen(5)  # 第三步：监听端口

print("waiting msg....")
receive_picture_conn, receive_picture_client_add = receive_picture.accept()  # 第四步：接收客户端的connect连接请求
send_msg_conn, send_msg_client_add = send_msg.accept()  # 第四步：接收客户端的connect连接请求

height = 960
width = 1280
fps = 30
threadLock = threading.Lock()
threads = []
send_data = [1.1111, 2.1231231, 3.1231, 4.12312, 5.123131, 6.123131, 7.1231232]

send_data_flag = 0
detect_img = 0
receive_img = 0
get_new_info = 0
class MyThread(threading.Thread):
    def __init__(self, thread_func, args, name):
        threading.Thread.__init__(self)
        # self.threadID = thread_id
        self.name = name
        self.func = thread_func
        self.args = args
        # self.counter = counter

    def get_result(self):
        return self.res

    def run(self):
        print("开启线程： " + self.name)
        # 获取锁，用于线程同步
        # threadLock.acquire()
        # 执行函数
        self.res = self.func(*self.args)
        # 释放锁，开启下一个线程
        # threadLock.release()


def float_to_bytes(f):
    bs = struct.pack("f", f)
    return bs
    # return (bs[3], bs[2], bs[1], bs[0])


send_data_byte = bytes(0)
for i in range(len(send_data)):
    send_data_byte_buffer = float_to_bytes(send_data[i])
    send_data_byte += send_data_byte_buffer


# send_thread = MyThread("send")
def send_msg():
    global send_data_flag,detect_img,receive_img,get_new_info
    print("send msg....")
    while True:
        try:
            # if detect_img == 0 and receive_img == 0 and get_new_info == 1:
            #     send_data_flag = 1
            send_msg_conn.send(send_data_byte)
                # send_data_flag = 0
                # get_new_info = 0
        except ConnectionResetError:
            break


image_to_yolo = np.zeros((height,width,3), np.uint8)
image_to_yolo.fill(255)
boxed_image =  np.zeros((height,width,3), np.uint8)
boxed_image.fill(255)
def disp_image():
    print("receive msg....")
    global image_to_yolo, boxed_image;
    global send_data_flag, detect_img, receive_img
    while True:
        try:
            data = receive_picture_conn.recv(4915200)  # 第五步：接收客户端传来的数据信息
            # data = struct.unpack('3686400i', data)
            # data = data.decode('utf-16')
            # print(data[2])
            # # data = data.split(b' ')
            # print(data[0])
            if len(data) == 4915200:
                # print("received msg length:", len(data))
                if send_data_flag == 0 and detect_img == 0:
                    receive_img = 1
                    img = Image.frombuffer('RGBA', (width, height), data)
                    image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                    image_to_yolo = cv2.flip(image, 0)
                    receive_img = 0
                    # cv2.imshow('1', boxed_image)
                    # cv2.waitKey(20)
                    # img.show()
                    # for index in range(len(data)):
                    #     data2 = int.from_bytes(data[index], byteorder="big", signed=False)
                    #     print(data2)
                    # rgbaimg = np.array([i for i in data])

                    # rgbaimg = np.array(data)  # .astype(np.uint8).reshape([1280, 720, 4])
                    # rgbaimg = ord(rgbaimg)
                    # print(rgbaimg.dtype)
                    # rgbaimg = rgbaimg.astype(np.uint8)
                    # print(rgbaimg)
                    # cv2.imshow("rgba", rgbaimg)
                    pass
                    # conn.send(data.upper())  # 第六步：给客户端回复信息
            else:
                break
        except ConnectionResetError:
            break


def yolo_detect():
    global image_to_yolo, boxed_image,get_new_info
    global send_data_flag, detect_img, receive_img
    yolo = YoloNet()
    yolo.load_data()
    # cc = cv2.Mat
    cc = np.zeros((height,width,3), np.uint8)
    cc.fill(255)
    while True:
        if send_data_flag == 0 and receive_img == 0:
            detect_img = 1
            cc = image_to_yolo.copy()
            detect_img = 0
        if len(cc) != 0:
            total_info, boxed_image = yolo.detect(cc)
            get_new_info = 1
            cv2.imshow('1', boxed_image)
            # cv2.waitKey(20)
        print(send_data_flag, detect_img, receive_img)
        cv2.waitKey(3)



receive_thread = MyThread(disp_image, (), "receive")
yolo_thread = MyThread(yolo_detect,(), "yolo")
send_thread = MyThread(send_msg, (), "send")

# 启动线程
yolo_thread.start()
receive_thread.start()
send_thread.start()

# 等待线程结束
receive_thread.join()
yolo_thread.join()
send_thread.join()

# while 1:
#     pass

# disp_image()
# try:
#     _thread.start_new_thread(disp_image, ())
# except():
#     print("error,start thread error!!")
#
# while 1:
#     pass
# 传输数据关闭
# conn.close()  # 第七步：传输数据关闭
