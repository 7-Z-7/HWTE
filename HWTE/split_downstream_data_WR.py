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

def leave():
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


  user_list=list(data.keys())
  # print(len(data[user_list[0]][4][1]))
  print(data[user_list[0]][4])
  print(len(data[user_list[1]][4][1]))
  print(len(data[user_list[2]][4][1]))
  print(len(data[user_list[3]][4][1]))
  # train_start=datetime.datetime(2020,1,1)
  # train_time_set=[datetime.datetime(2020,3,1)]
  train_time_set=[datetime.datetime(2019,12,1)]
  # eval_start=datetime.datetime(2020,7,1)
  # eval_time_set=[datetime.datetime(2020,4,1)]
  eval_time_set=[datetime.datetime(2020,12,1)]
  train_set=[]
  eval_set=[]
  train_set_num={'0':0,'1':0,'2':0}
  eval_set_num={'0':0,'1':0,'2':0}
  for one_user in user_list:
    for line in data[one_user]:#line[3]=performance, promotion, leave, leave_day,schedule
      if len(line[3][0])>0:
        performance=float(line[3][0])
        if line[0][0]in train_time_set:
          if performance<2.5:
            label='0'
            train_set_num[label]+=1
          elif performance>=2.5 and performance<=3.5:
            label='1'
            train_set_num[label] += 1
          else:
            label='2'
            train_set_num[label] += 1
          example=[line[0],line[1][-max_length:],line[2][-max_length:],label]
          train_set.append(example)
        elif line[0][0]in eval_time_set:
          if performance<2.5:
            label='0'
            eval_set_num[label]+=1
          elif performance>=2.5 and performance<=3.5:
            label='1'
            eval_set_num[label] += 1
          else:
            label='2'
            eval_set_num[label] += 1
          example=[line[0],line[1][-max_length:],line[2][-max_length:],label]
          eval_set.append(example)
  #sample for balance
  rate=1
  train_rate=(train_set_num['0']+train_set_num['2'])/train_set_num['1']
  eval_rate=(eval_set_num['0']+eval_set_num['2'])/eval_set_num['1']
  new_train_set=[]
  for one in train_set:
    if one[3]=='0' or one[3]=='2':
      new_train_set.append(one)
    else:
      if rng.random()<=train_rate*rate:
        new_train_set.append(one)
  new_eval_set=[]
  for one in eval_set:
    if one[3]=='0' or one[3]=='2':
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

def promotion():
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

def schedule():
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

def main(_):
  # leave()
  perf()
  # promotion()



if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
