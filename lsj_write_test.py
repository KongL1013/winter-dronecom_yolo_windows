import numpy as np
import pandas as pd


def tran_strid_int(str_id):
    int_id = 0
    if str_id == 'y':
        int_id = 0
    else:
        int_id = int(str_id)
    return int_id


def trans_check_data(data):
    if len(data) == 0:
        return [0, 0, 0]  # id_now,prop,left_x,left_y,right_x,right_y
    else:
        print("data", data)
        # print("type_data", type(data))
        id_now, pro, box = data
        id_now = tran_strid_int(id_now)
        left_x, left_y, right_x, right_y = box
        return [id_now, left_x, left_y, right_x, right_y]


send_current = [('y', 0.1, [1, 1, 1, 1]), ('1', 0.2, [2, 2, 2, 2]), ('1', 0.3, [3, 3, 3, 3]), ('2', 0.4, [4] * 4),
                ('2', 0.4, [5] * 4), ('2', 0.5, [6] * 4)]

if len(send_current) > 10:
    send_current = send_current[:10]
pd_data = pd.DataFrame(send_current, columns=['id', 'pro', 'box'])
for i in range(len(send_current)):
    pd_data.iat[i, 0] = tran_strid_int(pd_data.iat[i, 0])


data_count = dict(pd_data['id'].value_counts())


print(pd_data['id'].value_counts())
print(pd_data.iat[0, 0])
# pd_data = pd.DataFrame(send_current,index=[])
