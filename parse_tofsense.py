#!/usr/bin/env python3
# coding:utf-8
import serial
from struct import *
import time


def open_tofsense_serial():
    ser = serial.Serial("COM4", 115200, timeout=0.5)
    return ser, ser.isOpen()


# _access_data0 = chr(0x57).encode("utf-8") + chr(0x10).encode("utf-8") + chr(0xFF).encode("utf-8") + \
#                 chr(0xFF).encode("utf-8") + chr(0x00).encode("utf-8") + chr(0xFF).encode("utf-8") + \
#                 chr(0xFF).encode("utf-8") + chr(0x63).encode("utf-8")
# _access_data1 = chr(0x57) + chr(0x10) + chr(0xff) + chr(0xff) + chr(0x01) + chr(0xff) + chr(0xff) + chr(0x64)
# _access_data2 = chr(0x57) + chr(0x10) + chr(0xff) + chr(0xff) + chr(0x02) + chr(0xff) + chr(0xff) + chr(0x65)
# _access_data3 = chr(0x57) + chr(0x10) + chr(0xff) + chr(0xff) + chr(0x03) + chr(0xff) + chr(0xff) + chr(0x67)
# _access_data0 = pack('>8b', 87, 16, 255, 255, 0, 255, 255, 99)
# _access_data1 = pack('>8b', 87, 16, -128, -128, 1, -128, -128, 100)
# _access_data2 = pack('>8b', 87, 16, -128, -128, 2, -128, -128, 101)
# _access_data3 = pack('>8b', 87, 16, -128, -128, 3, -128, -128, 102)


def access_data(index, ser):
    _access_data0 = bytes.fromhex('5710FFFF00FFFF63')
    _access_data1 = bytes.fromhex('5710FFFF01FFFF64')
    _access_data2 = bytes.fromhex('5710FFFF02FFFF65')
    _access_data3 = bytes.fromhex('5710FFFF03FFFF66')
    if index == 0:
        ser.write(_access_data0)
    elif index == 1:
        ser.write(_access_data1)
    elif index == 2:
        ser.write(_access_data2)
    elif index == 3:
        ser.write(_access_data3)
    # print("sent data")


def receive_data(index, ser):
    system_time_bytes = b''
    dis_bytes = b''
    signal_strength_bytes = b''
    _databuff = ser.read()
    sum_chk = 0
    chk_status = False
    distance = 0
    dis_status = 0
    sensor_time = 0
    signal_strength = 0

    for times in range(15):
        if _databuff.hex() == '57':  # 检查是不是0x57
            break

    if _databuff.hex() == '57':  # 检查是不是0x57
        sum_chk += int.from_bytes(_databuff, byteorder='little', signed=False)
        _databuff = ser.read()
        sum_chk += int.from_bytes(_databuff, byteorder='little', signed=False)
        if _databuff.hex() == '00':  # 检查是不是0x00
            _databuff = ser.read()
            if _databuff.hex() == 'ff':
                sum_chk += int.from_bytes(_databuff, byteorder='little', signed=False)
                _databuff = ser.read()
                if int.from_bytes(_databuff, byteorder='little', signed=False) == index:
                    sum_chk += index
                    for i in range(4):
                        _buffer = ser.read()
                        system_time_bytes += _buffer
                        sum_chk += int.from_bytes(_buffer, byteorder='little', signed=False)

                    sensor_time = int.from_bytes(system_time_bytes, byteorder='little', signed=False)
                    for i in range(3):
                        _buffer = ser.read()
                        dis_bytes += _buffer
                        sum_chk += int.from_bytes(_buffer, byteorder='little', signed=False)

                    distance = (int.from_bytes(dis_bytes, byteorder='little', signed=False)) / 1000.0
                    dis_status_buffer = ser.read()
                    dis_status = int.from_bytes(dis_status_buffer, byteorder='little', signed=False)
                    sum_chk += int.from_bytes(dis_status_buffer, byteorder='little', signed=False)
                    for i in range(2):
                        _buffer = ser.read()
                        signal_strength_bytes += _buffer
                        sum_chk += int.from_bytes(_buffer, byteorder='little', signed=False)
                    signal_strength = int.from_bytes(signal_strength_bytes, byteorder='little', signed=False)
                    _databuff = ser.read()
                    if _databuff.hex() == 'ff':
                        sum_chk += int.from_bytes(_databuff, byteorder='little', signed=False)
                        _databuff = ser.read()
                        chk_status = (hex(sum_chk)[-2:] == _databuff.hex())
    if signal_strength < 1:
        distance = 99
    if (signal_strength == 1) & (dis_status != 0) & (dis_status != 1):
        distance = 99
    # else:
    #     if signal_strength < 1:
    #         distance = 99
    if chk_status:
        pass

        # print(index)
        # print(distance)
        # print(str(index) + " distance:" + str(distance) + " dis_status:" + str(dis_status) + " signal_strength:" + str(
            # signal_strength))
    else:
        print("error!!!!!!!! : "+str(index))

    # return chk_status, index, distance, dis_status, sensor_time, signal_strength
    return distance, sensor_time


# def test_receive():
#     global rbytes
#     for i in range(16):
#         rbytes += ser.read()
#     print(rbytes.hex())

if __name__ == '__main__':
    ser, ser_status = open_tofsense_serial()
    start_time = time.time()

    for i in range(100000):
        # access_data(0, ser)
        # receive_data(0, ser)
        access_data(1, ser)
        receive_data(1, ser)
        # access_data(2, ser)
        # receive_data(2, ser)
        # access_data(3, ser)
        # receive_data(3, ser)
        # print(i)
    end_time = time.time()
    print('totally cost time', end_time - start_time)

    # access_data(1)
    # receive_data()
    # access_data(2)
    # receive_data()
    # access_data(3)
    # receive_data()
    # print(_access_data1)
