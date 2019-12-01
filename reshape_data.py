import numpy as np
import pandas as pd

name = {'door': 1,'table':2,'doc':3,'pipe':4,'y':0}

def tran_strid_int(str_id):
    # int_id = 0
    # if str_id == 'y':
    #     int_id = 9
    # else:
    #     # int_id = int(str_id)
    #     int_id = 3  #11-13 night change

    return name[str_id]



def reshape_yolo_data(send_current):
    if len(send_current) == 0:
        return [0]
    if len(send_current) > 10:
        send_current = send_current[:10]
    pd_data = pd.DataFrame(send_current, columns=['id', 'pro', 'box'])
    for i in range(len(send_current)):
        pd_data.iat[i, 0] = tran_strid_int(pd_data.iat[i, 0])

    data_count = pd_data['id'].value_counts(sort=False) #sort according to counts number,from large to small
    send_yolo_info = []
    for i in range(len(data_count)):
        id_now = data_count.index[i]
        # print('id',id_now)
        id_cnt = data_count.values[i]
        index_find = np.where(pd_data['id'] == id_now)[0]

        for tt in range(id_cnt): ##tt loaction of id_cnt
            # print('tt',tt)
            send_id = list([id_now+id_cnt*0.1 + (tt+1)*0.01])
            # print('send_id',send_id)
            send_box = pd_data.iat[index_find[tt],2]
            # print('send_box',send_box)
            send_to = send_id+send_box
            # print('send_to',send_to)
            send_yolo_info.extend(send_to)
    return  send_yolo_info


##for chair start
    # cc = pd.DataFrame(send_current)
    # # print(cc)
    # index_chair = np.where(cc == 'y')
    # if len(index_chair[0]) != 0:
    #     # print(cc)
    #     index_np = np.array(index_chair)
    #     # print(index_np)
    #     # prop = cc[index_np[0][0],1]
    #     box_now = cc.iat[index_np[0][0], 2]
    #     left_x, left_y, right_x, right_y = box_now
    #     send_data_now = [1, left_x, left_y, right_x, right_y] + [0, 0]
    #     send_data_byte = trans_data_byte(send_data_now)
    # else:
    #     send_data_now = [0] * 7
    #     send_data_byte = trans_data_byte(send_data_now)

    ##for chair end


    ##laser test start
    # send_data_now = laser_current+detect_info
    # send_data_byte = trans_data_byte(send_data_now)
    ##laser test end


# def trans_check_data(data):
#     if len(data) == 0:
#         return [0, 0, 0]  # id_now,prop,left_x,left_y,right_x,right_y
#     else:
#         print("data", data)
#         # print("type_data", type(data))
#         id_now, pro, box = data
#         if id_now == 'chair':
#             id_now = 1
#         else:
#             id_now = 0
#         left_x, left_y, right_x, right_y = box
#         return [id_now, left_x, left_y, right_x, right_y]
#
#
# def tran_strid_int(str_id):
#     int_id = 0
#     if str_id == 'y':
#         int_id = 0
#     else:
#         int_id = int(str_id)
#     return int_id





if __name__ == '__main__':
    send_current = [('y', 0.1, [1, 1, 4, 1]), ('1', 0.2, [2, 2, 2, 2]), ('1', 0.3, [3, 3, 3, 3]), ('2', 0.4, [4] * 4),
                    ('2', 0.4, [5] * 4), ('2', 0.5, [6] * 4)]
    send_yolo_info = reshape_yolo_data(send_current)
    print(send_current)
    print(send_yolo_info)

##加上总长度补0