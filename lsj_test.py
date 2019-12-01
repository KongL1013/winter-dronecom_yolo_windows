from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet


netMain = None
metaMain = None
altNames = None


def load_data():
    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3.cfg"
    weightPath = "./yolo-weights/yolov3.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath) + "`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath) + "`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath) + "`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
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

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    total_info = []
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
        total_info.append((detection[0].decode(),detection[1] * 100,[xmin, ymin, xmax, ymax]))
    return img,total_info




def YOLO():

    load_data()
    cap = cv2.VideoCapture("ccani.mp4")
    cap.set(3, 1280)
    cap.set(4, 720)

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),#416
                                    darknet.network_height(netMain)),#416
                                   interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.50)  #thresh=0.50识别的阈值
        #detections=  (b'bird', 0.9903159141540527, (172.501708984375, 188.70071411132812, 56.11371994018555, 177.9843292236328))
        # nameTag, dets[j].prob[i], (b.x, b.y, b.w, b.h)

        # for i in range(len(detections)):
        #     print('detections= ',detections[i])

        image ,total_info= cvDrawBoxes(detections, frame_resized) #total_info [name ,pro,[左上，右下]]

        print('total_info=',total_info)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print('fps=',1/(time.time()-prev_time))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
    cap.release()

if __name__ == "__main__":
    YOLO()
