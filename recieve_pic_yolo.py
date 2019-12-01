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

import pandas as pd
import warnings
tiny = True

warnings.filterwarnings('ignore')

# 11-13晚上改动：识别最低阈值0.5->0.25，返回id都改成了3,去除9
laser_info = [0] * 4
last_laser_info = [0] * 4
send_data = [0] * 54
total_info = [('y', 0.9999, [1, 2, 3, 4])]
img_store = 15


class YoloNet:
    netMain = None
    metaMain = None
    altNames = None
    if tiny:
        configPath = "./cfg/yolov3-tiny.cfg"
        weightPath = "./backup/tinyfinal1125.weights"
        metaPath = "./cfg/voc.data"
    else:
        configPath = "./cfg/yolov3.cfg"
        weightPath = "./backup/full_final.weights"
        metaPath = "./cfg/voc.data"



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
        total_info_now = []
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
            total_info_now.append((detection[0].decode(), detection[1] * 100, [xmin, ymin, xmax, ymax]))
        return img, total_info_now

    def detect(self, image):
        # pre process pic
        # (rows, cols, channels) = image.shape
        # lower_blue = np.array([16, 0, 0])
        # upper_blue = np.array([44, 255, 255])
        #
        # getimg = np.zeros([rows, cols, channels], np.uint8)
        # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #
        # mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # image = cv2.add(getimg, image, mask=mask)
        # Create an image we reuse for each detect
        darknet_image = darknet.make_image(darknet.network_width(YoloNet.netMain),
                                           darknet.network_height(YoloNet.netMain), 3)
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(self.netMain),  # 416
                                    darknet.network_height(self.netMain)),  # 416
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(self.netMain, self.metaMain, darknet_image, thresh=0.5)  # thresh=0.50识别的阈值
        # print('detections = ',detections)
        # detections=  [(b'doc', 0.5034971833229065, (325.1002197265625, 331.2305603027344, 139.5221710205078, 157.04791259765625))]
        # nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)

        boxed_image, total_info_this = self.cvDrawBoxes(detections, frame_resized)  # total_info [name ,pro,[左上，右下]]


        # print('total_info=', total_info)
        boxed_image = cv2.cvtColor(boxed_image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Ddddemo', boxed_image)
        # cv2.waitKey(1)
        return total_info_this, boxed_image


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


def trans_data_byte(send_data):
    send_data_byte = bytes(0)
    for i in range(len(send_data)):
        send_data_byte_buffer = float_to_bytes(send_data[i])
        send_data_byte += send_data_byte_buffer
    return send_data_byte


from reshape_data import reshape_yolo_data


# send_thread = MyThread("send")
def send_msg():
    global total_info, laser_info, last_laser_info
    print("send msg....")
    while True:
        send_current = total_info.copy()
        laser_current = laser_info.copy()
        # detect_info 记得初始化
        yolo_data = reshape_yolo_data(send_current)

        send_data_piece = laser_current + yolo_data
        cou_shu = [0] * (54 - len(send_data_piece))
        send_data = send_data_piece + cou_shu
        send_data_byte = trans_data_byte(send_data)
        try:
            send_msg_conn.send(send_data_byte)
        except ConnectionResetError:
            print('send_msg crashed')
            break
        time.sleep(0.01)


image_to_yolo = np.zeros((height, width, 3), np.uint8)
image_to_yolo.fill(255)
boxed_image = np.zeros((height, width, 3), np.uint8)
boxed_image.fill(255)

crash_cnt = 0
def disp_image():
    print("receive msg....")
    global image_to_yolo, boxed_image,crash_cnt
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
                img = Image.frombuffer('RGBA', (width, height), data)
                image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                image_to_yolo = cv2.flip(image, 0)

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
                # pass
                # conn.send(data.upper())  # 第六步：给客户端回复信息
            else:
                print('break  disp_image  crashed = ',crash_cnt)
                crash_cnt += 1
                pass

        except ConnectionResetError:
            print('disp_image crashed')
            break

cnt_now = 0
save_path = r'D:\img_store\\'
def save_img(img):
    global cnt_now
    name = save_path + str(cnt_now) + '.jpg'
    cv2.imwrite(name,img)
    cnt_now += 1

def save_now():
    global image_to_yolo
    while True:
        try:
            cc = image_to_yolo.copy()
        except:
            print('*********************************fuck u twice************************')
        save_img(cc)
        cv2.waitKey(500)




def yolo_detect():
    global image_to_yolo, boxed_image, total_info

    yolo = YoloNet()
    yolo.load_data()
    cc = np.zeros((height, width, 3), np.uint8)
    cc.fill(255)
    cv2.namedWindow("show1",0)
    while True:
        try:
            cc = image_to_yolo.copy()
        except:
            print('*********************************fuck u************************')
        # cv2.imshow('tets',cc)
        total_info, boxed_image = yolo.detect(cc)
        # print('total_info',total_info)
        cv2.imshow('show1', boxed_image)
        cv2.waitKey(80)


from parse_tofsense import access_data
from parse_tofsense import receive_data
from parse_tofsense import open_tofsense_serial


def receive_laser_data():
    global laser_info
    global last_laser_info
    ser, is_ser_opened = open_tofsense_serial()
    if is_ser_opened:
        print('open COM succeed!!!')
    while is_ser_opened:
        try:
            access_data(0, ser)
            distance, sensor_time = receive_data(0, ser)

            # if abs(distance - last_laser_info[0]) <= 1:
            laser_info[0] = distance
            # elif distance < 3.0:
            #     laser_info[0] = distance
            # else:
            #     laser_info[0] = last_laser_info[0]
            #     print("0 jump detected " + str(distance) + "->" + str(last_laser_info[0]))
            last_laser_info[0] = distance

            access_data(1, ser)
            distance, sensor_time = receive_data(1, ser)
            # if abs(distance - last_laser_info[1]) <= 1:
            laser_info[1] = distance
            # elif distance < 3.0:
            #     laser_info[1] = distance
            # else:
            #     laser_info[1] = last_laser_info[1]
            #     print("1 jump detected " + str(distance) + "->" + str(last_laser_info[1]))
            # last_laser_info[1] = distance
            access_data(2, ser)
            distance, sensor_time = receive_data(2, ser)
            # if abs(distance - last_laser_info[2]) <= 1:
            laser_info[2] = distance
            # elif distance < 3.0:
            #     laser_info[2] = distance
            # else:
            #     laser_info[2] = last_laser_info[2]
            #     print("2 jump detected " + str(distance) + "->" + str(last_laser_info[2]))
            last_laser_info[2] = distance

            access_data(3, ser)
            distance, sensor_time = receive_data(3, ser)
            # if abs(distance - last_laser_info[3]) <= 1:
            laser_info[3] = distance
            # elif distance < 3.0:
            #     laser_info[3] = distance
            # else:
            #     laser_info[3] = last_laser_info[3]
            #     print("3 jump detected " + str(distance) + "->" + str(last_laser_info[3]))
            last_laser_info[3] = distance

        except:
            print('receive_laser_data error!!!!!!')

receive_thread = MyThread(disp_image, (), "receive")
yolo_thread = MyThread(yolo_detect, (), "yolo")
send_thread = MyThread(send_msg, (), "send")
receive_laser_thread = MyThread(receive_laser_data, (), "receive_laser_data")
save_thread = MyThread(save_now, (), "save")
# 启动线程
yolo_thread.start()
receive_thread.start()
send_thread.start()
receive_laser_thread.start()
save_thread.start()
# 等待线程结束
receive_thread.join()
yolo_thread.join()
send_thread.join()
receive_laser_thread.join()
save_thread.join()
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
