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
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("input_file_merge", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string("input_label", None,
                    "Input raw label dir (songxin wonderful_dataset).")

flags.DEFINE_string("input_promotion", None,
                    "Input raw promotion file (promotion).")

flags.DEFINE_string("input_EMP_INFO", None,
                    "Input raw F_EMP_INFO file.")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("id2uuap_file", "../../data/part-00000-8c3f2ac5-ec84-40e2-95b7-45bae51e5d8b-c000.csv",
                    "The id2uuap file for id transfer.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")

class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, segment_during, each_day_info,each_day_schedule, masked_lm_positions, masked_lm_labels,
               is_random_next,masked_lm_labels_during,rep_positions,rep_labels):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.segment_during=segment_during
    self.each_day_info=each_day_info
    self.each_day_schedule=each_day_schedule
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels
    self.masked_lm_labels_during=masked_lm_labels_during

    self.rep_positions=rep_positions
    self.rep_labels=rep_labels


  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "segment_during: %s\n" % (" ".join([str(x) for x in self.segment_during]))
    s += "each_day_info: %s\n" % (" ".join([str(x) for x in self.each_day_info]))
    s += "each_day_schedule: %s\n" % (" ".join([str(x) for x in self.each_day_schedule]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "masked_lm_labels_during: %s\n" % (" ".join(
      [tokenization.printable_text(str(x)) for x in self.masked_lm_labels_during]))
    s += "rep_positions: %s\n" % (" ".join(
      [tokenization.printable_text(str(x)) for x in self.rep_positions]))
    s += "rep_labels: %s\n" % (" ".join(
      [tokenization.printable_text(str(x)) for x in self.rep_labels]))
    s += "\n"
    return s

  def __repr__(self):
    return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature

def create_downstream_instances_DR(input_files,input_merge_files,input_label,input_promotion,input_EMP_INFO,tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
  """Create `TrainingInstance`s from raw text."""

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.

  #导入promotion
  # with open(input_promotion, 'r') as f:
  #   data = f.read()
  #   print(data)
  #   exit()

  #导入id2uuap
  id2uuap={}
  uuap2id={}
  with open(FLAGS.id2uuap_file, 'r') as f:
    data = f.read().split('\n')
    for line in data:
      split_line=line.split(',')
      if len(split_line)>=2:
        id2uuap[int(split_line[0])]=split_line[1]
        uuap2id[split_line[1]]=int(split_line[0])

  #导入label
  def label_load(input_label=None,load_type=1,pickle_path=None):
    """
    :param input_label:original_raw_data_dirpath
    :param load_type:
            1:only load from raw data
            2:load from raw data and save in pickle path
            3:only load from pickle path
    :param pickle_path: processed pickle path
    :return: user_label_dict
    """
    if load_type<=2:
      label_list=os.listdir(input_label)
      label_list.sort()
      label_list=[one for one in label_list if one>'2017-12-01']
      label_list=[one for one in label_list if one>'2017-12-01']
      print(label_list)
      user_label_dict={}
      for file_name in label_list:
        label_path=os.path.join(input_label,file_name)
        print(label_path)
        with open(label_path,'r') as f:
          data=f.read().split('\n')
          for line in data:
            split_line=line.split(',')
            if len(split_line)>1:
              user_name=int(split_line[0])
              day_time=datetime.datetime.strptime(split_line[1],'%Y-%m-%d')
              #leave['0','1']
              if len(split_line[3].strip())>0:
                leave='1'
              else:
                leave='0'
              #leave day float
              leave_day=split_line[2]
              #performance['1','2','3','4','5']
              if len(split_line[4].strip())>0:
                performance=split_line[4].strip()[0]
              else:
                performance=''
              #promotion['0','1']
              if len(split_line[5].strip())>0:
                promotion='1'
              else:
                promotion='0'
              if user_name in user_label_dict:
                user_label_dict[user_name].append([day_time,performance,promotion,leave,leave_day])
              else:
                user_label_dict[user_name] = [[day_time, performance, promotion, leave, leave_day]]
    if load_type==2:
      with open(pickle_path,'wb') as f:
        pickle.dump(user_label_dict,f)
    if load_type==3:
      with open(pickle_path, 'rb') as f:
        user_label_dict=pickle.load(f)
    return user_label_dict



  # 导入用户信息
  def info_load(input_EMP_INFO=None,load_type=1,pickle_path=None):
    """
    :param input_EMP_INFO:original_raw_data_dirpath
    :param load_type:
            1:only load from raw data
            2:load from raw data and save in pickle path
            3:only load from pickle path
    :param pickle_path: processed pickle path
    :return: user_info_dict
    """
    if load_type<=2:
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
      big_series_dict = {'研发': 1, '销售': 2, '直销体系': 2, '专业服务': 3, '市场及沟通': 4, '产品': 5, '管理类': 6, '销售服务': 7, '业务拓展': 8,
                         '政企行业解决方案和服务': 9, '未分类大序列': 0, '待定': 0, '物流': 0, '综合管理': 0}
      info_list = os.listdir(input_EMP_INFO)
      info_list.sort()
      info_list = [one for one in info_list if one > '20171231' and one<'20210101']
      user_info_dict = {}
      for file_name in info_list:
        info_path=os.path.join(input_EMP_INFO,file_name)
        print(info_path)
        with open(info_path,'r') as f:
          data=f.read().split('\n')
          for line in data:
            split_line=line.split(',')
            if len(split_line)>1:
              # print(split_line)
              level=split_line[25]+split_line[26]
              if level in sequence_dict:
                level_class=sequence_dict[level]
              else:
                print(level)
                level_class=0
              key_emp=split_line[27]
              if key_emp=='是':
                key_emp=1
              else:
                key_emp=0
              ##########新标签
              big_series=split_line[22]
              if big_series in big_series_dict:
                big_series_class=big_series_dict[big_series]
              else:
                big_series_class=0
              # exit()
              user_name=int(split_line[0])
              day_time=datetime.datetime.strptime(file_name,'%Y%m%d')
              if user_name in user_info_dict:
                # user_info_dict[user_name].append([day_time,level_class,key_emp])
                user_info_dict[user_name][day_time]=[level_class,key_emp,big_series_class]
              else:
                user_info_dict[user_name] = {day_time:[level_class,key_emp,big_series_class]}
    if load_type==2:
      with open(pickle_path, 'wb') as f:
        pickle.dump(user_info_dict, f)
    if load_type==3:
      with open(pickle_path, 'rb') as f:
        user_info_dict=pickle.load(f)
    return user_info_dict

  user_label_dict=label_load(input_label=input_label,load_type=3,pickle_path='user_label_dict.pickle')
  user_info_dict=info_load(input_EMP_INFO=input_EMP_INFO,load_type=3,pickle_path='user_info_multi_class_dict.pickle')
  # exit()

  user_list=list(user_label_dict.keys())
  print(user_label_dict[user_list[10]][1])
  user_list=list(user_info_dict.keys())
  day_list=list(user_info_dict[user_list[2]].keys())
  print(day_list)
  print(user_info_dict[user_list[2]][day_list[0]])
  # day_time, performance, promotion, leave, leave_day
  # level_class,key_emp
  perf={}
  prom={}
  leave={}
  level={}
  key={}
  big_series={}
  for user in user_label_dict:
    for one in user_label_dict[user]:
      one_day=one[0]
      one_perf=one[1]
      one_prom=one[2]
      one_leave=one[3]
      if one_perf in perf:
        perf[one_perf]+=1
      else:
        perf[one_perf] = 1
      if one_prom in prom:
        prom[one_prom] += 1
      else:
        prom[one_prom] = 1
      if one_leave in leave:
        leave[one_leave] += 1
      else:
        leave[one_leave] = 1
      if user in user_info_dict:
        if one_day in user_info_dict[user]:
          one_level=user_info_dict[user][one_day][0]
          one_key=user_info_dict[user][one_day][1]
          one_big_series=user_info_dict[user][one_day][2]
          if one_level in level:
            level[one_level] += 1
          else:
            level[one_level] = 1
          if one_key in key:
            key[one_key] += 1
          else:
            key[one_key] = 1
          if one_big_series in big_series:
            big_series[one_big_series] += 1
          else:
            big_series[one_big_series] = 1

  print('perf')
  print(perf)
  print('prom')
  print(prom)
  print('leave')
  print(leave)
  print('level')
  print(level)
  print('key')
  print(key)
  print('big_series')
  print(big_series)

  # exit()

  # for i in range(10):
  #   one=user_label_dict[list(user_label_dict.keys())[i]]
  #   one.sort(key=lambda x: x[0])
  #   for line in one:
  #     print(line)
  #   print('')
  #
  # for i in range(10):
  #   one=user_info_dict[list(user_info_dict.keys())[i]]
  #   one.sort(key=lambda x: x[0])
  #   for line in one:
  #     print(line)
  #   print('*****')

  # exit()

  all_WR={}
  for input_file in input_files:
    with open(input_file,'rb') as f:
      print('now_loading:{}'.format(input_file))
      one_file_all_WR=pickle.load(f)
      all_WR.update(one_file_all_WR)
      #{username: {day_time: [schedule_value, [[day_value, duration], ...], main_place], ...}, ...}
    # print(all_WR)

  #tiny_example_set
  # tiny_set={}
  # for i,user in enumerate(list(all_WR)):
  #   if i <5:
  #     tiny_set[user]=all_WR[user]
  # all_WR=tiny_set

  #去除空的一天
  print('user_num:{}'.format(len(all_WR)))
  local_num = 0
  for user in list(all_WR):
    for day in list(all_WR[user]):
      # 只保留训练的天数
      # if day >= 1577808000.0:  # (2020.1.1)
      # 只保留测试的天数
      # if day<1577808000.0:#(2020.1.1)
      if day>2608652800.0:#无限制
        del (all_WR[user][day])
      elif len(all_WR[user][day][1]) == 0:
        del (all_WR[user][day])
      else:
        local_num += len(all_WR[user][day][1])
    if len(all_WR[user]) <= 2:
      del (all_WR[user])

  print('local_num:{}'.format(local_num))

  all_WR_merge = {}
  for input_file in input_merge_files:
    with open(input_file, 'rb') as f:
      print('now_loading:{}'.format(input_file))
      one_file_all_WR = pickle.load(f)
      all_WR_merge.update(one_file_all_WR)
      # {username: {day_time: [schedule_value, [[day_value, duration], ...], main_place], ...}, ...}
    # print(all_WR)

  # 去除空的一天
  print('user_num_merge:{}'.format(len(all_WR_merge)))
  local_num = 0
  for user in list(all_WR_merge):
    for day in list(all_WR_merge[user]):
      # 只保留训练的天数
      # if day >= 1577808000.0:  # (2020.1.1)
      # 只保留测试的天数
      # if day<1577808000.0:#(2020.1.1)
      if day > 2608652800.0:  # 无限制
        del (all_WR_merge[user][day])
      elif len(all_WR_merge[user][day][1]) == 0:
        del (all_WR_merge[user][day])
      else:
        local_num += len(all_WR_merge[user][day][1])
    if len(all_WR_merge[user]) <= 2:
      del (all_WR_merge[user])

  print('local_num_merge:{}'.format(local_num))


  vocab_words = list(tokenizer.vocab.keys())
  print('vocab_size:{}'.format(len(vocab_words)))


  instances = {}
  num=0
  blank_num=0
  for user_index,user_name in enumerate(all_WR_merge):
    if user_index % 100==0:
      print(user_index)
    if user_name in uuap2id:
      if uuap2id[user_name] in user_label_dict:
        if user_name in all_WR:
          instances[uuap2id[user_name]]=create_ds_instances_from_DR(
                  all_WR_merge,all_WR, user_label_dict,user_info_dict,uuap2id,user_name,user_index, max_seq_length, short_seq_prob,
                  masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
          num += len(instances[uuap2id[user_name]])
        else:
          blank_num+=1
  print('instances_num:{}'.format(num))
  print(blank_num)
  return instances


def create_ds_instances_from_DR(
    all_WR,all_WR_common,user_label_dict,user_info_dict, uuap2id, user_name,user_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""
  WR=all_WR[user_name]
  WR_common=all_WR_common[user_name]
  labels=user_label_dict[uuap2id[user_name]]
  infos=user_info_dict[uuap2id[user_name]]

  # print(user_name)
  # print(WR)
  # exit()
  # document = all_documents[document_index]
  # print(document_index)
  # print(document)


  instances = []
  # start_time=datetime.datetime(2018,1,1)
  #only test set(no pretraining
  start_time=datetime.datetime(2018,2,1)
  end_time=datetime.datetime(2021,1,1)
  # print(labels)
  # print(labels[-1][0])
  if labels[-1][0]>=start_time:
    for one in labels:
      # print(one[0])
      if one[0]>=start_time:
        label_day = time.mktime(one[0].timetuple())
        # schedule=[0,0,0,0,0,0]
        schedule_list=[]
        day_position=[]
        main_place_list=[]
        guid=[one[0],uuap2id[user_name]]#day,id
        label=one[1:]#performance, promotion, leave, leave_day
        text_a_list=[]
        text_a_list_merge=[]
        text_a_during_list=[]
        text_a_during_list_merge=[]
        if one[0] in infos:
          info_label=infos[one[0]]#level_class,key_emp
        else:
          info_label=[]
        # for i in range(1,90):#day in one instances
        a_day=365
        if one[0] in [datetime.datetime(2018,12,1),datetime.datetime(2019,12,1),datetime.datetime(2020,12,1)]:
          a_day=365
        # print(one[0])
        # print(label_day)
        # print(WR)
        # print(WR_common)
        # print('----------')
        for i in range(1,a_day):#day in one instances
          if label_day-i*24*60*60+8*60*60 in WR_common:
            # print('WRCO')
            if label_day-i*24*60*60 in WR:
              # print('WR')
              one_day=WR[label_day - i*24 * 60 * 60]
              one_day_commom=WR_common[label_day - i*24 * 60 * 60+8*60*60]
              one_day_pair = one_day[1]
              one_day_pair_common = one_day_commom[1]

              # add final None tokens
              one_day_len = 288
              for one_token in one_day_pair_common:
                one_day_len -= one_token[1]
              if one_day_len > 0:
                one_day_pair_common.append(['None', one_day_len])
              if one_day_len>=0:
                one_day_pair_common.append(['[SEP]', 1])
              one_day_schedule = one_day[0]#performance, promotion, leave, leave_day,schedule%
              one_day_mainplace = one_day[2]

              one_day_len = 288
              for one_token in one_day_pair:
                one_day_len -= one_token[1]
              if one_day_len > 0:
                one_day_pair.append(['None', one_day_len])
              if one_day_len >= 0:
                one_day_pair.append(['[SEP]', 1])


              schedule_list.append(one_day_schedule)
              main_place_list.append(one_day_mainplace)
              day_position.append(i)
              text_a_list.append([x[0] for x in one_day_pair_common])
              text_a_list_merge.append([x[0] for x in one_day_pair])
              text_a_during_list.append([x[1] for x in one_day_pair_common])
              text_a_during_list_merge.append([x[1] for x in one_day_pair])
              # one_day_info = [user_Id, int(day_list[WR_index]), one_day_mainplace]

        instances.append([guid,text_a_list,text_a_during_list,label,info_label,schedule_list,main_place_list,day_position,text_a_list_merge,text_a_during_list_merge])#[guid,text_a,text_a_during,label]
        # exit()
  return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, tokens_during, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a wor d has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
        token.startswith("##")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  rng.shuffle(cand_indexes)

  output_tokens = list(tokens)
  output_tokens_during=list(tokens_during)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token
      output_tokens_during[index]=0

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  masked_lm_during_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)
    masked_lm_during_labels.append(tokens_during[p.index])

  return (output_tokens,output_tokens_during, masked_lm_positions, masked_lm_labels,masked_lm_during_labels)

def truncate_seq_pair_ori(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
    else:
      trunc_tokens.pop()

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
  """Truncates a pair of sequences to a maximum sequence length."""
  a_front=0
  a_back=0
  b_front=0
  b_back=0
  truncate_list=[a_front,a_back,b_front,b_back]
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_num_tokens:
      break

    trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
    trunc_index = 0 if len(tokens_a) > len(tokens_b) else 2
    assert len(trunc_tokens) >= 1

    # We want to sometimes truncate from the front and sometimes from the
    # back to add more randomness and avoid biases.
    if rng.random() < 0.5:
      del trunc_tokens[0]
      truncate_list[trunc_index]+=1
    else:
      trunc_tokens.pop()
      truncate_list[trunc_index+1] += 1
  return truncate_list

def input_files_root(path):
  # delta = 2
  # start_index = 0
  # end_index = 10
  # end_index = 1
  delta = 100  # 每次存储的原始文件数量
  start_index = 0  # 起始文件编号
  end_index = 1999  # 结束文件编号
  # end_index = 1999  # 结束文件编号
  input_files=[]
  for start in range(start_index, end_index, delta):
    # load_path = os.path.join(schedule_root, 'schedulepart-{}-{}'.format(start, start + delta))  # 存储的文件名
    # load_path = os.path.join('../../data/user_wordst10000-200000/', 'wordstpart-{}-{}'.format(start, start + delta))
    load_path = os.path.join(path, 'wordstpart-{}-{}'.format(start, start + delta))
    input_files.append(load_path)
  return input_files

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)#将 TensorFlow 日志信息输出到屏幕

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  # input_files = []
  # for input_pattern in FLAGS.input_file.split(","):
  #   print(input_pattern)
  #   input_files.extend(tf.gfile.Glob(input_pattern))
  # print(input_files)
  # exit()

  input_files=input_files_root(FLAGS.input_file)

  input_merge_files=input_files_root(FLAGS.input_file_merge)

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  for input_file_merge in input_merge_files:
    tf.logging.info("  %s", input_file_merge)

  rng = random.Random(FLAGS.random_seed)

  input_label=FLAGS.input_label
  input_promotion=FLAGS.input_promotion
  input_EMP_INFO=FLAGS.input_EMP_INFO

  # instances = create_training_instances_WR(
  #     input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
  #     FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
  #     rng)
  instances = create_downstream_instances_DR(
    input_files,input_merge_files,input_label,input_promotion,input_EMP_INFO, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
    FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
    rng)

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)
  # os.makedirs(output_files[0],exist_ok=True)
  path = os.path.join(output_files[0], 'all_dr_ds_instances_major_common.pickle')
  with open(path,'wb') as f:
    pickle.dump(instances,f)


  # write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
  #                                 FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("input_label")
  flags.mark_flag_as_required("input_promotion")
  flags.mark_flag_as_required("input_EMP_INFO")
  tf.app.run()
