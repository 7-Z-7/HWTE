# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import tokenization
import tensorflow as tf
import os
import pickle
import datetime
import time
import pandas as pd

flags = tf.flags

FLAGS = flags.FLAGS


def info_load(input_EMP_INFO=None, load_type=1, pickle_path=None):
  """
  :param input_EMP_INFO:original_raw_data_dirpath
  :param load_type:
          1:only load from raw data
          2:load from raw data and save in pickle path
          3:only load from pickle path
  :param pickle_path: processed pickle path
  :return: user_info_dict
  """
  if load_type <= 2:
    sequence_dict = {
      'Band3T1': 1, 'Band3T2': 1, 'Band3T3': 1, 'Band4T4': 1, 'Band4T5': 1, 'Band5T6': 1, 'Band5T7': 1,
      'Band5T8': 1, 'Band6T9': 1, 'Band6T10': 1, 'Band7T11': 1, 'Band7T12': 1,
      'Band3P1': 2, 'Band3P2': 2, 'Band3P3': 2, 'Band4P4': 2, 'Band4P5': 2, 'Band5P6': 2, 'Band5P7': 2,
      'Band5P8': 2, 'Band6P9': 2, 'Band6P10': 2, 'Band7P11': 2, 'Band7P12': 2,
      'Band3U1': 3, 'Band3U2': 3, 'Band3U3': 3, 'Band4U4': 3, 'Band4U5': 3, 'Band5U6': 3, 'Band5U7': 3,
      'Band5U8': 3, 'Band6U9': 3, 'Band6U10': 3, 'Band7U11': 3, 'Band7U12': 3,
      'Band1-': 4, 'Band2-': 4, 'Band3-': 4, 'Band4-': 4, 'Band3A': 4, 'Band3B': 4, 'Band4A': 4, 'Band4B': 4,
      'Band5A': 4, 'Band5B': 4, 'Band6A': 4, 'Band6B': 4, 'Band7-': 4,
      'M1A': 5, 'M1B': 5, 'M2A': 5, 'M2B': 5, 'M3A': 5, 'M3B': 5, 'M3C': 5, 'M4-': 5, 'M4A': 5, 'M4B': 5,
      'M5A': 5, 'M5B': 5, 'M6A': 5, 'M6B': 5, 'nullnull': 0, '--': 0, '未分类大层级未分类小层级': 0}
    big_series_num={'研发': 13122784, '销售': 66739, '直销体系': 7656414, '专业服务': 1633424, '市场及沟通': 371247, '产品': 5711400, '管理类': 1771015, '销售服务': 1658938, '未分类大序列': 594058, '业务拓展': 835256, '待定': 34373, '物流': 6646, '综合管理': 472, '政企行业解决方案和服务': 254395}
    big_series_dict={'研发': 1, '销售': 2, '直销体系': 2, '专业服务': 3, '市场及沟通': 4, '产品': 5, '管理类': 6, '销售服务': 7, '业务拓展': 8, '政企行业解决方案和服务': 9,'未分类大序列': 0, '待定': 0, '物流': 0, '综合管理': 0}
    info_list = os.listdir(input_EMP_INFO)
    info_list.sort()
    info_list = [one for one in info_list if one > '20171231' and one < '20210101']
    user_info_dict = {}
    group_count_dict={}
    type_count_dict={}
    big_series_count_dict={}
    for file_name in info_list:
      info_path = os.path.join(input_EMP_INFO, file_name)
      print(info_path)
      with open(info_path, 'r') as f:
        data = f.read().split('\n')
        for line in data:
          split_line = line.split(',')
          if len(split_line) > 1:
            # print(split_line)
            level = split_line[25] + split_line[26]
            group=split_line[3]
            type=split_line[6]
            big_series=split_line[22]
            if group in group_count_dict:
              group_count_dict[group]+=1
            else:
              group_count_dict[group] = 1
            if type in type_count_dict:
              type_count_dict[type]+=1
            else:
              type_count_dict[type] = 1
            if big_series in big_series_count_dict:
              big_series_count_dict[big_series]+=1
            else:
              big_series_count_dict[big_series] = 1

            if level in sequence_dict:
              level_class = sequence_dict[level]
            else:
              # print(level)
              level_class = 0
            key_emp = split_line[27]
            if key_emp == '是':
              key_emp = 1
            else:
              key_emp = 0
            # exit()
            user_name = int(split_line[0])
            day_time = datetime.datetime.strptime(file_name, '%Y%m%d')
            if user_name in user_info_dict:
              # user_info_dict[user_name].append([day_time,level_class,key_emp])
              user_info_dict[user_name][day_time] = [level_class, key_emp]
            else:
              user_info_dict[user_name] = {day_time: [level_class, key_emp]}
    print('--------------group:{}'.format(len(group_count_dict)))
    print(group_count_dict)
    print('--------------type:{}'.format(len(type_count_dict)))
    print(type_count_dict)
    print('--------------big_series:{}'.format(len(big_series_count_dict)))
    print(big_series_count_dict)

  return 0



info_load(input_EMP_INFO="../../data/F_EMP_INFO")
