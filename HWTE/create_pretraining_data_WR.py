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
import pandas as pd

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

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
    segment_during=list(instance.segment_during)
    assert len(input_ids) <= max_seq_length

    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)
      segment_during.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    masked_lm_weights = [1.0] * len(masked_lm_ids)
    masked_lm_labels_during=list(instance.masked_lm_labels_during)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)
      masked_lm_labels_during.append(0)

    next_sentence_label = 1 if instance.is_random_next else 0

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])
    features["segment_during"] = create_int_feature(segment_during)
    features["masked_lm_labels_during"] = create_int_feature(masked_lm_labels_during)
    # print("masked_lm_ids:{}".format(len(features["masked_lm_ids"].int64_list.value)))
    # print("segment_during:{}".format(len(features["segment_during"].int64_list.value)))
    # print("masked_lm_labels_during:{}".format(len(features["masked_lm_labels_during"].int64_list.value)))


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

def create_training_instances_WR(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
  """Create `TrainingInstance`s from raw text."""

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.
  all_WR={}
  for input_file in input_files:
    with open(input_file,'rb') as f:
      print('now_loading:{}'.format(input_file))
      one_file_all_WR=pickle.load(f)
      all_WR.update(one_file_all_WR)
      #{username: {day_time: [schedule_value, [[day_value, duration], ...], main_place], ...}, ...}
    # print(all_WR)
  #去除空的一天
  #tiny_example_set
  # tiny_set={}
  # for i,user in enumerate(list(all_WR)):
  #   if i <5:
  #     tiny_set[user]=all_WR[user]
  # all_WR=tiny_set


  print('user_num:{}'.format(len(all_WR)))
  local_num=0
  local_dict={}
  day_len=[]
  for user in list(all_WR):
    for day in list(all_WR[user]):
      #只保留训练的天数
      if day>=1577808000.0:#(2020.1.1)
      # if day>2608652800.0:#无限制
        del(all_WR[user][day])
      elif len(all_WR[user][day][1])==0:
        del(all_WR[user][day])
      else:
        local_num+=len(all_WR[user][day][1])
        day_len.append(len(all_WR[user][day][1]))
        for one in all_WR[user][day][1]:
          if one[0] in local_dict:
            local_dict[one[0]]+=1
          else:
            local_dict[one[0]]=1
    if len(all_WR[user])<=2:
      del(all_WR[user])

  print('local_num:{}'.format(local_num))
  with open('inform_of_traindata.pickle','wb') as f:
    pickle.dump([local_dict,day_len],f)
  print(max(day_len))
  print(local_dict)

  vocab_words = list(tokenizer.vocab.keys())
  print('vocab_size:{}'.format(len(vocab_words)))


  instances = []
  for _ in range(dupe_factor):
    for user_index,user_name in enumerate(all_WR):
      if user_index % 100==0:
        print(user_index)
      instances.extend(
          create_instances_from_WR(
              all_WR, user_name,user_index, max_seq_length, short_seq_prob,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

  rng.shuffle(instances)
  print('instances_num:{}'.format(len(instances)))
  return instances


def create_instances_from_WR(
    all_WR, user_name,user_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""
  WR=all_WR[user_name]
  # print(user_name)
  # print(WR)
  # exit()
  # document = all_documents[document_index]
  # print(document_index)
  # print(document)

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_info=[]
  current_length = 0
  WR_index = 0
  day_list=list(WR)
  user_list=list(all_WR)
  while WR_index < len(WR):
    # print('{}/{}'.format(str(WR_index),str(len(WR))))
    one_day = WR[day_list[WR_index]]
    one_day_pair=one_day[1]
    one_day_schedule=one_day[0]
    one_day_mainplace=one_day[2]
    current_chunk.append(one_day_pair)
    current_info.append([day_list[WR_index],one_day_schedule,one_day_mainplace])
    current_length += len(one_day_pair)
    if WR_index == len(WR) - 1 or current_length >= target_seq_length:
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        tokens_a_info=[]
        for j in range(a_end):
          info=current_info[j]+[len(tokens_a)]
          tokens_a_info.append(info)
          tokens_a.extend(current_chunk[j])

        tokens_b = []
        tokens_b_info=[]
        # Random next
        is_random_next = False
        if len(current_chunk) == 1 or rng.random() < 0.5:
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          random_user_index=0
          while True:
            for _ in range(10):
              random_user_index = rng.randint(0, len(all_WR) - 1)
              if random_user_index != user_index:
                break

            random_WR = all_WR[user_list[random_user_index]]
            random_start = rng.randint(0, len(random_WR) - 1)
            b_day_list = list(random_WR)
            for j in range(random_start, len(random_WR)):
              b_one_day=random_WR[b_day_list[j]]
              b_one_day_pair = b_one_day[1]
              b_one_day_schedule = b_one_day[0]
              b_one_day_mainplace = b_one_day[2]

              tokens_b_info.append([b_day_list[j], b_one_day_schedule, b_one_day_mainplace,len(tokens_b)])
              tokens_b.extend(b_one_day_pair)
              if len(tokens_b) >= target_b_length:
                break
            if len(tokens_b) >= target_b_length:
              break

          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste.
          num_unused_segments = len(current_chunk) - a_end
          WR_index -= num_unused_segments
        # Actual next
        else:
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            info=current_info[j]+[len(tokens_b)]
            tokens_b_info.append(info)
            tokens_b.extend(current_chunk[j])
        truncate_list=truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)
        # truncate_seq_pair_ori(tokens_a, tokens_b, max_num_tokens, rng)
        # print(truncate_list)
        # print(tokens_a)
        # print(tokens_a_info)
        # print(tokens_b)
        # print(tokens_b_info)
        # print(len(tokens_a))
        # print(len(tokens_b))

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        tokens = []
        user_ids = []
        tokens_during=[]
        tokens_info=[]
        tokens.append("[CLS]")
        tokens_during.append(0)
        user_ids.append(0)
        for token in tokens_a:
          tokens.append(token[0])
          tokens_during.append(token[1])
          user_ids.append(0)

        end_a=len(tokens)
        for i in range(len(tokens_a_info)):
          new_position=tokens_a_info[i][3]+1-truncate_list[0]
          if new_position<=0:
            if len(tokens_info)>0:
              del(tokens_info[0])
              tokens_a_info[i][3] = new_position
              tokens_info.append(tokens_a_info[i])
            else:
              tokens_a_info[i][3]=new_position
              tokens_info.append(tokens_a_info[i])
          elif new_position>end_a:
            pass
          else:
            tokens_a_info[i][3] = new_position
            tokens_info.append(tokens_a_info[i])


        tokens.append("[SEP]")
        tokens_during.append(1)
        user_ids.append(0)

        start_b=len(tokens)
        a_info_len=len(tokens_info)
        for token in tokens_b:
          tokens.append(token[0])
          tokens_during.append(token[1])
          user_ids.append(1)

        end_b = len(tokens)
        for i in range(len(tokens_b_info)):
          new_position=tokens_b_info[i][3]+start_b-truncate_list[2]
          if new_position<start_b:
            if len(tokens_info)>a_info_len:
              del(tokens_info[a_info_len])
              tokens_b_info[i][3] = new_position
              tokens_info.append(tokens_b_info[i])
            else:
              tokens_b_info[i][3]=new_position
              tokens_info.append(tokens_b_info[i])
          elif new_position>end_b:
            pass
          else:
            tokens_b_info[i][3] = new_position
            tokens_info.append(tokens_b_info[i])

        tokens.append("[SEP]")
        tokens_during.append(1)
        user_ids.append(1)

        for i,one in enumerate(tokens):
          tokens[i]=str(one)
        # print(len(tokens))
        # print(len(tokens_info))
        # print(len(user_ids))
        # print(tokens)
        # print(tokens_info)
        # print(user_ids)


        (tokens,tokens_during, masked_lm_positions,
         masked_lm_labels,masked_lm_labels_during) = create_masked_lm_predictions(
             tokens,tokens_during, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
        # print(tokens)
        # print(tokens_during)
        # print(masked_lm_positions)
        # print(masked_lm_labels)
        # print(masked_lm_labels_during)

        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=user_ids,
            segment_during=tokens_during,
            each_day_info=tokens_info,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels,
            masked_lm_labels_during=masked_lm_labels_during)
        # print(instance)
        # if random.randint(0, 10) < 2:
        #   exit()
        instances.append(instance)
      current_chunk = []
      current_length = 0
    WR_index += 1
  return instances


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, tokens_during, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective."""

  cand_indexes = []
  for (i, token) in enumerate(tokens):
    # if token == "[CLS]" or token == "[SEP]":
    if token == "[CLS]" or token == "[SEP]" or token =="None":
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
      # output_tokens_during[index]=0
      output_tokens_during[index]=tokens_during[index]

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

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances_WR(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng)

  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)

def vocab_file_create(load_path,save_path):
  df=pd.read_csv(load_path)
  vocab=list(set(list(df['format_number'])))
  vocab.sort()
  vocab=['None']+vocab
  for i,one in enumerate(vocab):
    vocab[i]=str(one)

  print(vocab)
  with open(save_path,'w') as f:
    f.writelines('[PAD]\n')
    for i in range(1,100):
      f.writelines('[unused{}]\n'.format(i))
    f.writelines('[UNK]\n')
    f.writelines('[CLS]\n')
    f.writelines('[SEP]\n')
    f.writelines('[MASK]\n')
    for i in range(104,106):
      f.writelines('[unused{}]\n'.format(i))
    f.writelines('None\n')
    for i in range(107,109):
      f.writelines('[unused{}]\n'.format(i))
    f.writelines('0\n')
    for i in range(110, 112):
      f.writelines('[unused{}]\n'.format(i))
    for one in vocab:
      if one =='0' or one=='None':
        pass
      else:
        f.writelines(one)
        f.writelines('\n')

# def read_test(path):





if __name__ == "__main__":
  # vocab_file_create("../../data/ap_stamp.csv","ap_vocab.txt")
  # read_test('./tf_during_train_data.tfrecord')
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()
