# coding: utf-8

import numpy as np
file_path = "/Users/ZhangXuemiao/Desktop/dialog-data/ubuntu/val.dat"

count = -1

max_context_len = -1
max_response_len = -1
vocabulay_len = -1

with open(file_path, 'r', encoding='utf-8') as f:
    for record in f.readlines():
        splits = record.split(';')
        if len(splits) != 4:
            continue
        id = int(splits[0])
        label = int(splits[3])

        context_ids_str = splits[1].split(' ')
        context_ids_len = len(context_ids_str)
        # if context_ids_len > max_context_len:
        #     max_context_len = context_ids_len
        # context_ids = np.zeros(context_ids_len)
        for i in range(context_ids_len):
            location = int(context_ids_str[i])
            if location > vocabulay_len:
                vocabulay_len = location
                print(vocabulay_len)


        response_ids_str = splits[2].split(' ')
        response_ids_len = len(response_ids_str)
        # if response_ids_len > max_response_len:
        #     max_response_len = response_ids_len
        for i in range(response_ids_len):
            res_location = int(response_ids_str[i])
            if res_location > vocabulay_len:
                vocabulay_len = res_location
                print(vocabulay_len)

    print('final ->', vocabulay_len)
        # response_ids = np.zeros(response_ids_len)
