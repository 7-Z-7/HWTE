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
from dateutil.relativedelta import relativedelta

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input lines example file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output lines example file (multi_label).")

flags.DEFINE_string(
    "split", None,
    "Output lines example file (multi_label).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("id2uuap_file", "../../data/part-00000-8c3f2ac5-ec84-40e2-95b7-45bae51e5d8b-c000.csv",
                    "The id2uuap file for id transfer.")


flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")


flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")


class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, segment_during, each_day_info, masked_lm_positions, masked_lm_labels,
               is_random_next,masked_lm_labels_during):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.segment_during=segment_during
    self.each_day_info=each_day_info
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels
    self.masked_lm_labels_during=masked_lm_labels_during

  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "segment_during: %s\n" % (" ".join([str(x) for x in self.segment_during]))
    s += "each_day_info: %s\n" % (" ".join([str(x) for x in self.each_day_info]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "masked_lm_labels_during: %s\n" % (" ".join(
      [tokenization.printable_text(str(x)) for x in self.masked_lm_labels_during]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()

def leave_old():
  tf.logging.set_verbosity(tf.logging.INFO)#将 TensorFlow 日志信息输出到屏幕

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  # input_files = []
  # for input_pattern in FLAGS.input_file.split(","):
  #   print(input_pattern)
  #   input_files.extend(tf.gfile.Glob(input_pattern))
  # print(input_files)
  # exit()

  input_file=FLAGS.input_file
  output_file=FLAGS.output_file

  tf.logging.info("*** Reading from input files ***")
  tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)

  with open(input_file,'rb') as f:
    data=pickle.load(f)

  max_length=FLAGS.max_seq_length-3

  #time split_leav 1 month
  delta_month=31
  user_list=list(data.keys())
  # print(len(data[user_list[0]][4][1]))
  print(data[user_list[0]][4])
  print(len(data[user_list[1]][4][1]))
  print(len(data[user_list[2]][4][1]))
  print(len(data[user_list[3]][4][1]))
  # train_start=datetime.datetime(2020,1,1)
  # train_time_set=[datetime.datetime(2020,3,1)]
  train_time_set=[datetime.datetime(2020,6,1)]
  # eval_start=datetime.datetime(2020,7,1)
  # eval_time_set=[datetime.datetime(2020,4,1)]
  eval_time_set=[datetime.datetime(2020,7,1)]
  train_set=[]
  eval_set=[]
  train_set_num={'0':0,'1':0}
  eval_set_num={'0':0,'1':0}
  for one_user in user_list:
    for line in data[one_user]:#performance, promotion, leave, leave_day,schedule
      try:
        if len(line[3][3].strip())>0:
          leave_day=float(line[3][3])
        else:
          leave_day=-1
      except:
        print([line[3][3]])
      if line[0][0]in train_time_set:
        if leave_day>0 and leave_day<=delta_month:
          label='1'
          train_set_num[label]+=1
        else:
          label='0'
          train_set_num[label] += 1
        example=[line[0],line[1][-max_length:],line[2][-max_length:],label]
        train_set.append(example)
      elif line[0][0]in eval_time_set:
        if leave_day>0 and leave_day<=delta_month:
          label='1'
          eval_set_num[label]+=1
        else:
          label='0'
          eval_set_num[label] += 1
        example=[line[0],line[1][-max_length:],line[2][-max_length:],label]
        eval_set.append(example)
  #sample for balance
  rate=1
  train_rate=train_set_num['1']/train_set_num['0']
  eval_rate=eval_set_num['1']/eval_set_num['0']
  new_train_set=[]
  for one in train_set:
    if one[3]=='1':
      new_train_set.append(one)
    else:
      if rng.random()<=train_rate*rate:
        new_train_set.append(one)
  new_eval_set=[]
  for one in eval_set:
    if one[3]=='1':
      new_eval_set.append(one)
    else:
      if rng.random()<=eval_rate*rate:
        new_eval_set.append(one)
  train_set=new_train_set
  eval_set=new_eval_set
  print(train_set_num)
  print(eval_set_num)

  os.makedirs(output_file,exist_ok=True)
  train_path=os.path.join(output_file,'train.pickle')
  eval_path=os.path.join(output_file,'eval.pickle')
  with open(train_path,'wb') as f:
    pickle.dump(train_set,f)
    print(len(train_set))
  with open(eval_path, 'wb') as f:
    pickle.dump(eval_set, f)
    print(len(eval_set))

  exit()

  # instances = create_training_instances_WR(
  #     input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
  #     FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
  #     rng)

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)
  # os.makedirs(output_files[0],exist_ok=True)
  path = os.path.join(output_files[0], 'all_ds_instances.pickle')
  with open(path,'wb') as f:
    pass
    # pickle.dump(instances,f)


  # write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
  #                                 FLAGS.max_predictions_per_seq, output_files)

def perf():
  #4，5是低绩效，1，2是高绩效(1,2->0, 4,5->1)
  #不使用3的数据

  tf.logging.set_verbosity(tf.logging.INFO)#将 TensorFlow 日志信息输出到屏幕

  # input_files = []
  # for input_pattern in FLAGS.input_file.split(","):
  #   print(input_pattern)
  #   input_files.extend(tf.gfile.Glob(input_pattern))
  # print(input_files)
  # exit()

  input_file=FLAGS.input_file
  output_file=FLAGS.output_file

  tf.logging.info("*** Reading from input files ***")
  tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)

  with open(input_file,'rb') as f:
    data=pickle.load(f)

  user_list=list(data.keys())
  # print(len(data[user_list[0]][4][1]))
  print(data[user_list[0]][4])
  print(len(data[user_list[1]][4][1]))
  print(len(data[user_list[2]][4][1]))
  print(len(data[user_list[3]][4][1]))
  # train_start=datetime.datetime(2020,1,1)
  # train_time_set=[datetime.datetime(2020,3,1)]
  # train_time_set=[datetime.datetime(2019,12,1),datetime.datetime(2018,12,1)]
  train_time_set=[datetime.datetime(2018,12,1)]
  # eval_start=datetime.datetime(2020,7,1)
  # eval_time_set=[datetime.datetime(2020,4,1)]
  # eval_time_set=[datetime.datetime(2020,12,1)]
  eval_time_set=[datetime.datetime(2019,12,1)]

  train_set=[]
  eval_set=[]
  train_set_num={'0':0,'1':0}
  eval_set_num={'0':0,'1':0}
  for one_user in user_list:
    for line in data[one_user]:
      #line[0]=day_time,id
      #line[3]=performance, promotion, leave, leave_day,schedule
      #line[4]=level,key
      #line[5]=schedule_list
      #line[6]=main_place_list
      #line[7]=day_position
      if len(line[3][0])>0:
        performance=float(line[3][0])
        if line[0][0]in train_time_set:
          if performance<2.5:
            label='0'
            train_set_num[label]+=1
          elif performance>=2.5 and performance<=3.5:
            continue
            # label='1'
            # train_set_num[label] += 1
          else:
            label='1'
            train_set_num[label] += 1
          example=[line[0],line[1],line[2],line[7],label]
          train_set.append(example)
        elif line[0][0]in eval_time_set:
          if performance<2.5:
            label='0'
            eval_set_num[label]+=1
          elif performance>=2.5 and performance<=3.5:
            continue
            # label='1'
            # eval_set_num[label] += 1
          else:
            label='1'
            eval_set_num[label] += 1
          example=[line[0],line[1],line[2],line[7],label]
          eval_set.append(example)
  #sample for balance
  # rate=1
  # train_rate=(train_set_num['1']/train_set_num['0'])
  # eval_rate=(eval_set_num['1']/eval_set_num['0'])
  # new_train_set=[]
  # for one in train_set:
  #   if one[3]=='1':
  #     new_train_set.append(one)
  #   else:
  #     if rng.random()<=train_rate*rate:
  #       new_train_set.append(one)
  # new_eval_set=[]
  # for one in eval_set:
  #   if one[3]=='1':
  #     new_eval_set.append(one)
  #   else:
  #     if rng.random()<=eval_rate*rate:
  #       new_eval_set.append(one)
  # train_set=new_train_set
  # eval_set=new_eval_set
  print(train_set_num)
  print(eval_set_num)

  os.makedirs(output_file,exist_ok=True)
  train_path=os.path.join(output_file,'train.pickle')
  eval_path=os.path.join(output_file,'eval.pickle')
  with open(train_path,'wb') as f:
    pickle.dump(train_set,f)
    print(len(train_set))
  with open(eval_path, 'wb') as f:
    pickle.dump(eval_set, f)
    print(len(eval_set))

  # exit()

def perf_one_class(type="excellent"):
  #excellent：高绩效分类
  #poor：低绩效分类
  #4，5是低绩效，1，2是高绩效(1,2->0, 4,5->1)
  #不使用3的数据

  tf.logging.set_verbosity(tf.logging.INFO)#将 TensorFlow 日志信息输出到屏幕

  # input_files = []
  # for input_pattern in FLAGS.input_file.split(","):
  #   print(input_pattern)
  #   input_files.extend(tf.gfile.Glob(input_pattern))
  # print(input_files)
  # exit()

  input_file=FLAGS.input_file
  output_file=FLAGS.output_file

  tf.logging.info("*** Reading from input files ***")
  tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)

  with open(input_file,'rb') as f:
    data=pickle.load(f)

  user_list=list(data.keys())
  # print(len(data[user_list[0]][4][1]))
  print(data[user_list[0]][4])
  print(len(data[user_list[1]][4][1]))
  print(len(data[user_list[2]][4][1]))
  print(len(data[user_list[3]][4][1]))
  # train_start=datetime.datetime(2020,1,1)
  # train_time_set=[datetime.datetime(2020,3,1)]
  # train_time_set=[datetime.datetime(2019,12,1),datetime.datetime(2018,12,1)]
  train_time_set=[datetime.datetime(2018,12,1)]
  # eval_start=datetime.datetime(2020,7,1)
  # eval_time_set=[datetime.datetime(2020,4,1)]
  # eval_time_set=[datetime.datetime(2020,12,1)]
  eval_time_set=[datetime.datetime(2019,12,1)]
  train_set=[]
  eval_set=[]
  train_set_num={'0':0,'1':0}
  eval_set_num={'0':0,'1':0}
  for one_user in user_list:
    for line in data[one_user]:
      #line[0]=day_time,id
      #line[3]=performance, promotion, leave, leave_day,schedule
      #line[4]=level,key
      #line[5]=schedule_list
      #line[6]=main_place_list
      #line[7]=day_position
      def performance_label(performance,type):
          if type=="excellent":
              if performance<2.5:
                  label='1'
              else:
                  label='0'
          elif type=="poor":
              if performance>3.5:
                  label='1'
              else:
                  label='0'
          else:
              raise ValueError('type error')
          return label

      if len(line[3][0])>0:
        performance=float(line[3][0])
        if line[0][0]in train_time_set:
          label=performance_label(performance,type)
          if label=='-1':
              continue
          train_set_num[label] += 1
          example=[line[0],line[1],line[2],line[7],label]
          train_set.append(example)
        elif line[0][0]in eval_time_set:
          label = performance_label(performance, type)
          if label == '-1':
              continue
          train_set_num[label] += 1
          example=[line[0],line[1],line[2],line[7],label]
          eval_set.append(example)
  #sample for balance
  # rate=1
  # train_rate=(train_set_num['1']/train_set_num['0'])
  # eval_rate=(eval_set_num['1']/eval_set_num['0'])
  # new_train_set=[]
  # for one in train_set:
  #   if one[3]=='1':
  #     new_train_set.append(one)
  #   else:
  #     if rng.random()<=train_rate*rate:
  #       new_train_set.append(one)
  # new_eval_set=[]
  # for one in eval_set:
  #   if one[3]=='1':
  #     new_eval_set.append(one)
  #   else:
  #     if rng.random()<=eval_rate*rate:
  #       new_eval_set.append(one)
  # train_set=new_train_set
  # eval_set=new_eval_set
  print(train_set_num)
  print(eval_set_num)

  os.makedirs(output_file,exist_ok=True)
  train_path=os.path.join(output_file,'train.pickle')
  eval_path=os.path.join(output_file,'eval.pickle')
  with open(train_path,'wb') as f:
    pickle.dump(train_set,f)
    print(len(train_set))
  with open(eval_path, 'wb') as f:
    pickle.dump(eval_set, f)
    print(len(eval_set))

  # exit()

def leave(pre_month=1,his_month=1):
  #pre_month 预测数个月内是否会离职（1个月，3个月）
  #his_month 预测使用的可见历史月数（1个月，12个月）
  #0未离职，1离职

  tf.logging.set_verbosity(tf.logging.INFO)#将 TensorFlow 日志信息输出到屏幕

  # input_files = []
  # for input_pattern in FLAGS.input_file.split(","):
  #   print(input_pattern)
  #   input_files.extend(tf.gfile.Glob(input_pattern))
  # print(input_files)
  # exit()

  input_file=FLAGS.input_file
  output_file=FLAGS.output_file

  tf.logging.info("*** Reading from input files ***")
  tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)

  with open(input_file,'rb') as f:
    data=pickle.load(f)

  user_list=list(data.keys())
  # print(len(data[user_list[0]][4][1]))
  print(data[user_list[0]][4])
  print(len(data[user_list[1]][4][1]))
  print(len(data[user_list[2]][4][1]))
  print(len(data[user_list[3]][4][1]))
  # train_start=datetime.datetime(2020,1,1)
  # train_time_set=[datetime.datetime(2020,3,1)]
  # train_time_set=[datetime.datetime(2019,12,1),datetime.datetime(2018,12,1)]
  datetime.timedelta()
  train_time_set=[datetime.datetime(2020,1,1)+relativedelta(months=1)*i for i in range(6)]
  # eval_start=datetime.datetime(2020,7,1)
  # eval_time_set=[datetime.datetime(2020,4,1)]
  eval_time_set=[datetime.datetime(2020,12,1)-relativedelta(months=1)*i for i in range(7-pre_month)]
  train_set=[]
  eval_set=[]
  train_set_num={'0':0,'1':0}
  eval_set_num={'0':0,'1':0}
  for one_user in user_list:
    for line in data[one_user]:
      #line[0]=day_time,id
      #line[1]=text_a_list#倒序每天tokens（8.1,7.31,7.30,7.29...）
      #line[2]=text_a_during_list
      #line[3]=performance, promotion, leave, leave_day,schedule
      #line[4]=level,key
      #line[5]=schedule_list
      #line[6]=main_place_list
      #line[7]=day_position
      # print(line[7])
      for i,position in enumerate(line[7]):
        if position>30*his_month:
          break
      tokens_list=line[1][:i]
      durings_list=line[2][:i]
      positions_list=line[7][:i]
      label='0'
      if len(line[3][3])>0:
        leave_day=float(line[3][3])
        if leave_day>0 and leave_day<=pre_month*30:
          label='1'
      if line[0][0]in train_time_set:
        train_set_num[label]+=1
        example=[line[0],tokens_list,durings_list,positions_list,label]
        train_set.append(example)
      elif line[0][0]in eval_time_set:
        eval_set_num[label]+=1
        example=[line[0],tokens_list,durings_list,positions_list,label]
        eval_set.append(example)
  # sample for balance
  rate=1
  train_rate=(train_set_num['1']/train_set_num['0'])
  eval_rate=(eval_set_num['1']/eval_set_num['0'])
  new_train_set=[]
  for one in train_set:
    if one[4]=='1':
      new_train_set.append(one)
    else:
      if rng.random()<=train_rate*rate:
        new_train_set.append(one)
  new_eval_set=[]
  for one in eval_set:
    if one[4]=='1':
      new_eval_set.append(one)
    else:
      if rng.random()<=eval_rate*rate:
        new_eval_set.append(one)
  train_set=new_train_set
  eval_set=new_eval_set
  print(train_set_num)
  print(eval_set_num)

  os.makedirs(output_file,exist_ok=True)
  train_path=os.path.join(output_file,'train.pickle')
  eval_path=os.path.join(output_file,'eval.pickle')
  with open(train_path,'wb') as f:
    pickle.dump(train_set,f)
    print(len(train_set))
  with open(eval_path, 'wb') as f:
    pickle.dump(eval_set, f)
    print(len(eval_set))

def leave_near(pre_month=1,his_month=1):
  #pre_month 预测数个月内是否会离职（1个月，3个月）
  #his_month 预测使用的可见历史月数（1个月，12个月）
  #0未离职，1离职

  tf.logging.set_verbosity(tf.logging.INFO)#将 TensorFlow 日志信息输出到屏幕

  # input_files = []
  # for input_pattern in FLAGS.input_file.split(","):
  #   print(input_pattern)
  #   input_files.extend(tf.gfile.Glob(input_pattern))
  # print(input_files)
  # exit()

  input_file=FLAGS.input_file
  output_file=FLAGS.output_file

  tf.logging.info("*** Reading from input files ***")
  tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)

  with open(input_file,'rb') as f:
    data=pickle.load(f)

  user_list=list(data.keys())
  # print(len(data[user_list[0]][4][1]))
  print(data[user_list[0]][4])
  print(len(data[user_list[1]][4][1]))
  print(len(data[user_list[2]][4][1]))
  print(len(data[user_list[3]][4][1]))
  # train_start=datetime.datetime(2020,1,1)
  # train_time_set=[datetime.datetime(2020,3,1)]
  # train_time_set=[datetime.datetime(2019,12,1),datetime.datetime(2018,12,1)]
  datetime.timedelta()
  train_time_set=[datetime.datetime(2020,1,1)+relativedelta(months=1)*i for i in range(6)]
  # eval_start=datetime.datetime(2020,7,1)
  # eval_time_set=[datetime.datetime(2020,4,1)]
  eval_time_set=[datetime.datetime(2020,12,1)-relativedelta(months=1)*i for i in range(7-pre_month)]
  train_set=[]
  eval_set=[]
  train_set_num={'0':0,'1':0}
  eval_set_num={'0':0,'1':0}
  for one_user in user_list:
    for line in data[one_user]:
      #line[0]=day_time,id
      #line[1]=text_a_list#倒序每天tokens（8.1,7.31,7.30,7.29...）
      #line[2]=text_a_during_list
      #line[3]=performance, promotion, leave, leave_day,schedule
      #line[4]=level,key
      #line[5]=schedule_list
      #line[6]=main_place_list
      #line[7]=day_position
      # print(line[7])
      for i,position in enumerate(line[7]):
        if position>30*his_month:
          break
      tokens_list=line[1][:i]
      durings_list=line[2][:i]
      positions_list=line[7][:i]
      main_place_list=line[6][-20:]
      if len(main_place_list)<1:
          continue
      # print('--------------')
      # print(line[6])
      main_place=max(main_place_list, key=main_place_list.count)
      label='0'
      if len(line[3][3])>0:
        leave_day=float(line[3][3])
        if leave_day>0 and leave_day<=pre_month*30:
          label='1'
      if line[0][0]in train_time_set:
        # train_set_num[label]+=1
        example=[line[0],tokens_list,durings_list,positions_list,label,main_place]
        train_set.append(example)
      elif line[0][0]in eval_time_set:
        # eval_set_num[label]+=1
        example=[line[0],tokens_list,durings_list,positions_list,label,main_place]
        eval_set.append(example)
  #del not near leave
  train_leave_palce = []
  eval_leave_palce = []
  for one in train_set:
      if one[4] == '1':
          train_leave_palce.append(one[-1])
  for one in eval_set:
      if one[4] == '1':
          eval_leave_palce.append(one[-1])
  new_train_set=[]
  new_eval_set=[]
  for one in train_set:
    if one[-1] in train_leave_palce:
        new_train_set.append(one[:-1])
        train_set_num[one[4]]+=1
  for one in eval_set:
    #测试集也选择了near的数据
    if one[-1] in eval_leave_palce:
        new_eval_set.append(one[:-1])
        eval_set_num[one[4]]+=1
    #测试集为单纯采样
    # new_eval_set.append(one[:-1])
    # eval_set_num[one[4]] += 1
  train_set=new_train_set
  eval_set=new_eval_set
  # sample for balance
  rate=3
  train_rate=(train_set_num['1']/train_set_num['0'])
  eval_rate=(eval_set_num['1']/eval_set_num['0'])
  new_train_set=[]
  for one in train_set:
    if one[4]=='1':
      new_train_set.append(one)
    else:
      if rng.random()<=train_rate*rate:
        new_train_set.append(one)
  new_eval_set=[]
  for one in eval_set:
    if one[4]=='1':
      new_eval_set.append(one)
    else:
      if rng.random()<=eval_rate*rate:
        new_eval_set.append(one)
  train_set=new_train_set
  eval_set=new_eval_set
  print(train_set_num)
  print(eval_set_num)

  os.makedirs(output_file,exist_ok=True)
  train_path=os.path.join(output_file,'train.pickle')
  eval_path=os.path.join(output_file,'eval.pickle')
  with open(train_path,'wb') as f:
    pickle.dump(train_set,f)
    print(len(train_set))
  with open(eval_path, 'wb') as f:
    pickle.dump(eval_set, f)
    print(len(eval_set))


def level(his_month=1,train_rate=0.5,type=3):
  #预测职级,按照人分
  #his_month 预测使用的可见历史月数（1个月，12个月）
  #1:T，2:P, 3:U, 4:B 5:M
  #type 3:TPU
  #type 5:TPUBM

  tf.logging.set_verbosity(tf.logging.INFO)#将 TensorFlow 日志信息输出到屏幕

  # input_files = []
  # for input_pattern in FLAGS.input_file.split(","):
  #   print(input_pattern)
  #   input_files.extend(tf.gfile.Glob(input_pattern))
  # print(input_files)
  # exit()

  input_file=FLAGS.input_file
  output_file=FLAGS.output_file

  tf.logging.info("*** Reading from input files ***")
  tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)

  with open(input_file,'rb') as f:
    data=pickle.load(f)

  user_list=list(data.keys())
  # print(len(data[user_list[0]][4][1]))
  print(data[user_list[0]][4])
  print(len(data[user_list[1]][4][1]))
  print(len(data[user_list[2]][4][1]))
  print(len(data[user_list[3]][4][1]))
  # train_start=datetime.datetime(2020,1,1)
  # train_time_set=[datetime.datetime(2020,3,1)]
  # train_time_set=[datetime.datetime(2019,12,1),datetime.datetime(2018,12,1)]
  datetime.timedelta()
  train_time_set=[datetime.datetime(2020,1,1)+relativedelta(months=1)*i for i in range(6)]
  # eval_start=datetime.datetime(2020,7,1)
  # eval_time_set=[datetime.datetime(2020,4,1)]
  eval_time_set=[datetime.datetime(2020,12,1)-relativedelta(months=1)*i for i in range(6)]
  train_set=[]
  eval_set=[]
  train_set_num={'1':0,'2':0,'3':0,'4':0,'5':0}
  eval_set_num={'1':0,'2':0,'3':0,'4':0,'5':0}
  for one_user in user_list:
    if rng.random()<=train_rate:
      train_flag=True
    else:
      train_flag=False
    for line in data[one_user]:
      #line[0]=day_time,id
      #line[1]=text_a_list#倒序每天tokens（8.1,7.31,7.30,7.29...）
      #line[2]=text_a_during_list
      #line[3]=performance, promotion, leave, leave_day,schedule
      #line[4]=level,key
      #line[5]=schedule_list
      #line[6]=main_place_list
      #line[7]=day_position
      # print(line[7])
      if len(line[4])>0:
        label=line[4][0]
      else:
        continue
      if label <=0:
        continue
      if type==3:
        if label>=4:
          continue
      elif type==5:
        pass

      label=str(label)
      i=0
      for i,position in enumerate(line[7]):
        if position>30*his_month:
          break
      if i>0:
        tokens_list=line[1][:i]
        durings_list=line[2][:i]
        positions_list=line[7][:i]
        if line[0][0]in train_time_set:
          if train_flag:
            train_set_num[label]+=1
            example=[line[0],tokens_list,durings_list,positions_list,label]
            train_set.append(example)
        elif line[0][0]in eval_time_set:
          if train_flag==False:
            eval_set_num[label]+=1
            example=[line[0],tokens_list,durings_list,positions_list,label]
            eval_set.append(example)
  # sample for balance
  # rate=1
  # train_rate=(train_set_num['1']/train_set_num['0'])
  # eval_rate=(eval_set_num['1']/eval_set_num['0'])
  # new_train_set=[]
  # for one in train_set:
  #   if one[4]=='1':
  #     new_train_set.append(one)
  #   else:
  #     if rng.random()<=train_rate*rate:
  #       new_train_set.append(one)
  # new_eval_set=[]
  # for one in eval_set:
  #   if one[4]=='1':
  #     new_eval_set.append(one)
  #   else:
  #     if rng.random()<=eval_rate*rate:
  #       new_eval_set.append(one)
  # train_set=new_train_set
  # eval_set=new_eval_set
  print(train_set_num)
  print(eval_set_num)

  os.makedirs(output_file,exist_ok=True)
  train_path=os.path.join(output_file,'train.pickle')
  eval_path=os.path.join(output_file,'eval.pickle')
  with open(train_path,'wb') as f:
    pickle.dump(train_set,f)
    print(len(train_set))
  with open(eval_path, 'wb') as f:
    pickle.dump(eval_set, f)
    print(len(eval_set))

  # exit()

def big_series(his_month=1,train_rate=0.5,type=3):
  #预测职级,按照人分
  #his_month 预测使用的可见历史月数（1个月，12个月）
  #1:T，2:P, 3:U, 4:B 5:M
  #type 3:TPU
  #type 5:TPUBM

  tf.logging.set_verbosity(tf.logging.INFO)#将 TensorFlow 日志信息输出到屏幕

  # input_files = []
  # for input_pattern in FLAGS.input_file.split(","):
  #   print(input_pattern)
  #   input_files.extend(tf.gfile.Glob(input_pattern))
  # print(input_files)
  # exit()

  input_file=FLAGS.input_file
  output_file=FLAGS.output_file

  tf.logging.info("*** Reading from input files ***")
  tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)

  with open(input_file,'rb') as f:
    data=pickle.load(f)

  user_list=list(data.keys())
  # print(len(data[user_list[0]][4][1]))
  print(data[user_list[0]][4])
  print(len(data[user_list[1]][4][1]))
  print(len(data[user_list[2]][4][1]))
  print(len(data[user_list[3]][4][1]))
  # train_start=datetime.datetime(2020,1,1)
  # train_time_set=[datetime.datetime(2020,3,1)]
  # train_time_set=[datetime.datetime(2019,12,1),datetime.datetime(2018,12,1)]
  datetime.timedelta()
  train_time_set=[datetime.datetime(2020,1,1)+relativedelta(months=1)*i for i in range(6)]
  # eval_start=datetime.datetime(2020,7,1)
  # eval_time_set=[datetime.datetime(2020,4,1)]
  eval_time_set=[datetime.datetime(2020,12,1)-relativedelta(months=1)*i for i in range(6)]
  train_set=[]
  eval_set=[]
  train_set_num={'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0}
  eval_set_num={'0':0,'1':0,'2':0,'3':0,'4':0,'5':0,'6':0,'7':0,'8':0,'9':0}
  for one_user in user_list:
    if rng.random()<=train_rate:
      train_flag=True
    else:
      train_flag=False
    for line in data[one_user]:
      #line[0]=day_time,id
      #line[1]=text_a_list#倒序每天tokens（8.1,7.31,7.30,7.29...）
      #line[2]=text_a_during_list
      #line[3]=performance, promotion, leave, leave_day,schedule
      #line[4]=level,key
      #line[5]=schedule_list
      #line[6]=main_place_list
      #line[7]=day_position
      # print(line[7])
      if len(line[4])>0:
        label=line[4][2]
      else:
        continue
      if type==8:
        if label==0:
          continue
        if label==2:
          continue
      # if label <=0:
      #   continue
      # if type==3:
      #   if label>=4:
      #     continue
      # elif type==5:
      #   pass

      label=str(label)
      i=0
      for i,position in enumerate(line[7]):
        if position>30*his_month:
          break
      if i>0:
        tokens_list=line[1][:i]
        durings_list=line[2][:i]
        positions_list=line[7][:i]
        if line[0][0]in train_time_set:
          if train_flag:
            train_set_num[label]+=1
            example=[line[0],tokens_list,durings_list,positions_list,label]
            train_set.append(example)
        elif line[0][0]in eval_time_set:
          if train_flag==False:
            eval_set_num[label]+=1
            example=[line[0],tokens_list,durings_list,positions_list,label]
            eval_set.append(example)
  # sample for balance
  # rate=1
  # train_rate=(train_set_num['1']/train_set_num['0'])
  # eval_rate=(eval_set_num['1']/eval_set_num['0'])
  # new_train_set=[]
  # for one in train_set:
  #   if one[4]=='1':
  #     new_train_set.append(one)
  #   else:
  #     if rng.random()<=train_rate*rate:
  #       new_train_set.append(one)
  # new_eval_set=[]
  # for one in eval_set:
  #   if one[4]=='1':
  #     new_eval_set.append(one)
  #   else:
  #     if rng.random()<=eval_rate*rate:
  #       new_eval_set.append(one)
  # train_set=new_train_set
  # eval_set=new_eval_set
  print(train_set_num)
  print(eval_set_num)

  os.makedirs(output_file,exist_ok=True)
  train_path=os.path.join(output_file,'train.pickle')
  eval_path=os.path.join(output_file,'eval.pickle')
  with open(train_path,'wb') as f:
    pickle.dump(train_set,f)
    print(len(train_set))
  with open(eval_path, 'wb') as f:
    pickle.dump(eval_set, f)
    print(len(eval_set))

  # exit()


def next_prediction(his_day=28,pre_hour=1):
  #预测接下来x小时用户位置（排序预测

  tf.logging.set_verbosity(tf.logging.INFO)#将 TensorFlow 日志信息输出到屏幕

  # input_files = []
  # for input_pattern in FLAGS.input_file.split(","):
  #   print(input_pattern)
  #   input_files.extend(tf.gfile.Glob(input_pattern))
  # print(input_files)
  # exit()

  input_file=FLAGS.input_file
  output_file=FLAGS.output_file

  tf.logging.info("*** Reading from input files ***")
  tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)

  with open(input_file,'rb') as f:
    data=pickle.load(f)

  user_list=list(data.keys())
  # print(len(data[user_list[0]][4][1]))
  print(data[user_list[0]][4])
  print(len(data[user_list[1]][4][1]))
  print(len(data[user_list[2]][4][1]))
  print(len(data[user_list[3]][4][1]))
  # train_start=datetime.datetime(2020,1,1)
  # train_time_set=[datetime.datetime(2020,3,1)]
  # train_time_set=[datetime.datetime(2019,12,1),datetime.datetime(2018,12,1)]
  datetime.timedelta()
  train_time_set=[datetime.datetime(2020,1,1)+relativedelta(months=1)*i for i in range(6)]
  # eval_start=datetime.datetime(2020,7,1)
  # eval_time_set=[datetime.datetime(2020,4,1)]
  eval_time_set=[datetime.datetime(2020,12,1)-relativedelta(months=1)*i for i in range(6)]
  train_set=[]
  eval_set=[]
  train_set_num={'1':0,'2':0,'3':0,'4':0,'5':0}
  eval_set_num={'1':0,'2':0,'3':0,'4':0,'5':0}
  for one_user in user_list:
    for line in data[one_user]:
      #line[0]=day_time,id
      #line[1]=text_a_list#倒序每天tokens（8.1,7.31,7.30,7.29...）
      #line[2]=text_a_during_list
      #line[3]=performance, promotion, leave, leave_day,schedule
      #line[4]=level,key
      #line[5]=schedule_list
      #line[6]=main_place_list
      #line[7]=day_position
      # print(line[7])


      if line[0][0]in train_time_set or line[0][0]in eval_time_set:
        for predict_day in range(1,30):
          tokens_list = []
          durings_list = []
          positions_list = []
          if predict_day in line[7]:
            for i, position in enumerate(line[7]):
              if position >= predict_day and position<=predict_day+his_day:
                tokens_list.append(line[1][i])
                durings_list.append(line[2][i])
                positions_list.append(line[7][i]+1-predict_day)
            first_day=tokens_list[0]
            first_during=durings_list[0]
            if len(first_day)>10:
              for i,token in enumerate(first_day):
                if i >2 and i<len(first_day)-3:
                  new_first_day=first_day[:i]
                  new_first_during=first_during[:i]
                  pre_during=0
                  pre_dict={}
                  for j in range(i+1,len(first_day)):
                    if pre_during<pre_hour*12:
                      pre_during+=first_during[j]
                      if first_day[j] in pre_dict:
                        pre_dict[first_day[j]]+=first_during[j]
                      else:
                        pre_dict[first_day[j]]=first_during[j]
                    else:
                      break
                  sorted_pre=sorted(pre_dict.items(), key=lambda kv: kv[1],reverse=True)
                  token_label,during_label=zip(*sorted_pre)
                  new_tokens_list=[new_first_day]+tokens_list[1:]
                  new_durings_list=[new_first_during]+durings_list[1:]
                  example = [line[0], new_tokens_list, new_durings_list, positions_list, token_label,during_label]
                  if line[0][0]in train_time_set:
                    train_set.append(example)
                  else:
                    eval_set.append(example)
  # sample for balance
  # rate=1
  # train_rate=(train_set_num['1']/train_set_num['0'])
  # eval_rate=(eval_set_num['1']/eval_set_num['0'])
  # new_train_set=[]
  # for one in train_set:
  #   if one[4]=='1':
  #     new_train_set.append(one)
  #   else:
  #     if rng.random()<=train_rate*rate:
  #       new_train_set.append(one)
  # new_eval_set=[]
  # for one in eval_set:
  #   if one[4]=='1':
  #     new_eval_set.append(one)
  #   else:
  #     if rng.random()<=eval_rate*rate:
  #       new_eval_set.append(one)
  # train_set=new_train_set
  # eval_set=new_eval_set
  print(train_set_num)
  print(eval_set_num)

  os.makedirs(output_file,exist_ok=True)
  train_path=os.path.join(output_file,'train.pickle')
  eval_path=os.path.join(output_file,'eval.pickle')
  with open(train_path,'wb') as f:
    pickle.dump(train_set,f)
    print(len(train_set))
  with open(eval_path, 'wb') as f:
    pickle.dump(eval_set, f)
    print(len(eval_set))

  # exit()

def next_prediction_common(his_day=28,pre_hour=1):
  #预测接下来x小时用户位置（排序预测

  tf.logging.set_verbosity(tf.logging.INFO)#将 TensorFlow 日志信息输出到屏幕

  # input_files = []
  # for input_pattern in FLAGS.input_file.split(","):
  #   print(input_pattern)
  #   input_files.extend(tf.gfile.Glob(input_pattern))
  # print(input_files)
  # exit()

  input_file=FLAGS.input_file
  output_file=FLAGS.output_file

  tf.logging.info("*** Reading from input files ***")
  tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)

  with open(input_file,'rb') as f:
    data=pickle.load(f)

  user_list=list(data.keys())
  # print(len(data[user_list[0]][4][1]))
  print(data[user_list[0]][4])
  print(len(data[user_list[1]][4][1]))
  print(len(data[user_list[2]][4][1]))
  print(len(data[user_list[3]][4][1]))
  # train_start=datetime.datetime(2020,1,1)
  # train_time_set=[datetime.datetime(2020,3,1)]
  # train_time_set=[datetime.datetime(2019,12,1),datetime.datetime(2018,12,1)]
  datetime.timedelta()
  train_time_set=[datetime.datetime(2020,1,1)+relativedelta(months=1)*i for i in range(6)]
  # eval_start=datetime.datetime(2020,7,1)
  # eval_time_set=[datetime.datetime(2020,4,1)]
  eval_time_set=[datetime.datetime(2020,12,1)-relativedelta(months=1)*i for i in range(6)]
  train_set=[]
  eval_set=[]
  train_set_num={'1':0,'2':0,'3':0,'4':0,'5':0}
  eval_set_num={'1':0,'2':0,'3':0,'4':0,'5':0}
  for one_user in user_list:
    for line in data[one_user]:
      #line[0]=day_time,id
      #line[1]=text_a_list#倒序每天tokens（8.1,7.31,7.30,7.29...）
      #line[2]=text_a_during_list
      #line[3]=performance, promotion, leave, leave_day,schedule
      #line[4]=level,key
      #line[5]=schedule_list
      #line[6]=main_place_list
      #line[7]=day_position
      # print(line[7])


      if line[0][0]in train_time_set or line[0][0]in eval_time_set:
        for predict_day in range(1,30):
          tokens_list = []
          tokens_merge_list = []
          durings_list = []
          durings_merge_list = []
          positions_list = []
          if predict_day in line[7]:
            for i, position in enumerate(line[7]):
              if position >= predict_day and position<=predict_day+his_day:
                tokens_list.append(line[1][i])
                tokens_merge_list.append(line[8][i])
                durings_list.append(line[2][i])
                durings_merge_list.append(line[9][i])
                positions_list.append(line[7][i]+1-predict_day)
            first_day=tokens_list[0]
            first_merge_day=tokens_merge_list[0]
            first_during=durings_list[0]
            first_merge_during=durings_merge_list[0]
            if len(first_day)>10:
              for i,token in enumerate(first_day):
                if i >2 and i<len(first_day)-3:
                  new_first_day=first_day[:i]
                  new_first_during=first_during[:i]
                  sum_during=sum(new_first_during)

                  merge_time=0
                  check_list=[]
                  check_during_list=[]
                  for k,merge_one in enumerate(first_merge_day):
                    if merge_time+first_merge_during[k]>sum_during:
                      check_list=first_merge_day[k:]
                      check_during_list=first_merge_during[k:]
                      check_during_list[0]-=sum_during-merge_time
                      break
                    else:
                      merge_time+=first_merge_during[k]

                  pre_during=0
                  pre_dict={}
                  for j in range(len(check_list)):
                    if pre_during<pre_hour*12:
                      pre_during+=check_during_list[j]
                      if check_list[j] in pre_dict:
                        pre_dict[check_list[j]]+=check_during_list[j]
                      else:
                        pre_dict[check_list[j]]=check_during_list[j]
                    else:
                      break
                  sorted_pre=sorted(pre_dict.items(), key=lambda kv: kv[1],reverse=True)
                  token_label,during_label=zip(*sorted_pre)
                  new_tokens_list=[new_first_day]+tokens_list[1:]
                  new_durings_list=[new_first_during]+durings_list[1:]
                  example = [line[0], new_tokens_list, new_durings_list, positions_list, token_label,during_label]
                  if line[0][0]in train_time_set:
                    train_set.append(example)
                  else:
                    eval_set.append(example)
  # sample for balance
  # rate=1
  # train_rate=(train_set_num['1']/train_set_num['0'])
  # eval_rate=(eval_set_num['1']/eval_set_num['0'])
  # new_train_set=[]
  # for one in train_set:
  #   if one[4]=='1':
  #     new_train_set.append(one)
  #   else:
  #     if rng.random()<=train_rate*rate:
  #       new_train_set.append(one)
  # new_eval_set=[]
  # for one in eval_set:
  #   if one[4]=='1':
  #     new_eval_set.append(one)
  #   else:
  #     if rng.random()<=eval_rate*rate:
  #       new_eval_set.append(one)
  # train_set=new_train_set
  # eval_set=new_eval_set
  print(train_set_num)
  print(eval_set_num)

  os.makedirs(output_file,exist_ok=True)
  train_path=os.path.join(output_file,'train.pickle')
  eval_path=os.path.join(output_file,'eval.pickle')
  with open(train_path,'wb') as f:
    pickle.dump(train_set,f)
    print(len(train_set))
  with open(eval_path, 'wb') as f:
    pickle.dump(eval_set, f)
    print(len(eval_set))

  # exit()

def schedule():
  #估计当天用户的行为分布（outofwork, working，meeting）

  tf.logging.set_verbosity(tf.logging.INFO)#将 TensorFlow 日志信息输出到屏幕

  # input_files = []
  # for input_pattern in FLAGS.input_file.split(","):
  #   print(input_pattern)
  #   input_files.extend(tf.gfile.Glob(input_pattern))
  # print(input_files)
  # exit()

  input_file=FLAGS.input_file
  output_file=FLAGS.output_file

  tf.logging.info("*** Reading from input files ***")
  tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)

  with open(input_file,'rb') as f:
    data=pickle.load(f)

  user_list=list(data.keys())
  # print(len(data[user_list[0]][4][1]))
  print(data[user_list[0]][4])
  print(len(data[user_list[1]][4][1]))
  print(len(data[user_list[2]][4][1]))
  print(len(data[user_list[3]][4][1]))
  # train_start=datetime.datetime(2020,1,1)
  # train_time_set=[datetime.datetime(2020,3,1)]
  # train_time_set=[datetime.datetime(2019,12,1),datetime.datetime(2018,12,1)]
  datetime.timedelta()
  # train_time_set=[datetime.datetime(2020,1,1)+relativedelta(months=1)*i for i in range(6)]
  train_time_set=[datetime.datetime(2020,6,1)]
  # eval_start=datetime.datetime(2020,7,1)
  # eval_time_set=[datetime.datetime(2020,4,1)]
  # eval_time_set=[datetime.datetime(2020,12,1)-relativedelta(months=1)*i for i in range(6)]
  eval_time_set=[datetime.datetime(2020,7,1)]
  train_set=[]
  eval_set=[]
  train_set_num={'1':0,'2':0,'3':0,'4':0}
  eval_set_num={'1':0,'2':0,'3':0,'4':0}
  for one_user in user_list:
    for line in data[one_user]:
      #line[0]=day_time,id
      #line[1]=text_a_list#倒序每天tokens（8.1,7.31,7.30,7.29...）
      #line[2]=text_a_during_list
      #line[3]=performance, promotion, leave, leave_day,schedule
      #line[4]=level,key
      #line[5]=schedule_list# 'home','desk','meeting','moving','others','unknown'
      #line[6]=main_place_list
      #line[7]=day_position
      # print(line[7])

      if line[0][0]in train_time_set or line[0][0]in eval_time_set:
        for i, position in enumerate(line[7]):
          if position<30 and len(line[1])>10:
            tokens=line[1][i]
            durings=line[2][i]
            #outofwork, deskworking，meeting,other
            print(line[5][i])
            label=line[5][i][:3]+[sum(line[5][i][3:])]
            example = [line[0], tokens, durings, label]
            if line[0][0]in train_time_set:
              for k,one in enumerate(label):
                train_set_num[str(k+1)]+=one
              train_set.append(example)
            else:
              for k,one in enumerate(label):
                eval_set_num[str(k+1)]+=one
              eval_set.append(example)
  # sample for balance
  # rate=1
  # train_rate=(train_set_num['1']/train_set_num['0'])
  # eval_rate=(eval_set_num['1']/eval_set_num['0'])
  # new_train_set=[]
  # for one in train_set:
  #   if one[4]=='1':
  #     new_train_set.append(one)
  #   else:
  #     if rng.random()<=train_rate*rate:
  #       new_train_set.append(one)
  # new_eval_set=[]
  # for one in eval_set:
  #   if one[4]=='1':
  #     new_eval_set.append(one)
  #   else:
  #     if rng.random()<=eval_rate*rate:
  #       new_eval_set.append(one)
  # train_set=new_train_set
  # eval_set=new_eval_set
  print(train_set_num)
  print(eval_set_num)

  os.makedirs(output_file,exist_ok=True)
  train_path=os.path.join(output_file,'train.pickle')
  eval_path=os.path.join(output_file,'eval.pickle')
  with open(train_path,'wb') as f:
    pickle.dump(train_set,f)
    print(len(train_set))
  with open(eval_path, 'wb') as f:
    pickle.dump(eval_set, f)
    print(len(eval_set))

  # exit()

def main(_):
  #promotion()

  # leave(pre_month=1,his_month=12)#使用x个月的数据预测下x月是否会离职
  # perf()#使用一年数据预测绩效
  # level(his_month=12,train_rate=0.5,type=5)#使用x个月数据预测职业类别
  next_prediction(his_day=28,pre_hour=1)#使用x天数据预测下一个小时的位置
  # next_prediction_common(his_day=28,pre_hour=1)#使用x天数据预测下一个小时的位置
  # schedule()#使用一天的数据预测当日工作时间，会议时间和非办公时间

  # big_series(his_month=12,train_rate=0.5,type=10)#使用x个月数据预测职业类别
  # big_series(his_month=12,train_rate=0.5,type=8)#使用x个月数据预测职业类别


  # leave_near(pre_month=1,his_month=12)#使用x个月的数据预测下x月是否会离职,训练数据采样时取附近的
  # perf_one_class(type="excellent")#使用一年数据预测绩效(poor,excellent)
  # perf_one_class(type="poor")#使用一年数据预测绩效(poor,excellent)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
