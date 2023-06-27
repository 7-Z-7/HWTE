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
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling_DR as modeling
import optimization
import tokenization
import tensorflow as tf
import pickle
import random
import tf_metrics


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "gpu_device", '2',
    "The index of gpu device.")

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "stop_gradient", False,
    "stop gradient for parameters from pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_day_length", 128,
    "The maximum total input day length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("gradient_accumulation_multiplier", 1, "gradient accumulation multiplier for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_bool("sin_position", False, "use sin position or trainable position.")

flags.DEFINE_float("data_sampling_rate", 1, "sampling rate for data load.")

flags.DEFINE_float("binary_threshold", -1, "threshold for binary classification.")

flags.DEFINE_float("random_seed", 1234, "random seed")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class MultiDaysInputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, token_list, during_list, positions=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      token_list: string. The untokenized text of the day sequence list.
      during_list: string. The during of untokenized text of the day sequence list.
      position_list: (Optional) string. The position of day
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.token_list = token_list
    self.during_list= during_list
    self.positions= positions
    self.label = label

class DayInputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, tokens, durings, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      tokens: string. The untokenized session of one day sequence.
      durings: string. The during of untokenized session of one day sequence.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.tokens = tokens
    self.durings= durings
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class MultiDaysInputFeatures(object):
  """A single set of features of data."""
  def __init__(self,
               input_ids_list,
               input_mask_list,
               during_list,
               positions,
               label_ids):
    self.input_ids_list = input_ids_list
    self.input_mask_list = input_mask_list
    self.during_list = during_list
    self.positions = positions
    self.label_ids = label_ids


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir,rate,rng):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir,rate,rng):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir,rate,rng):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  def _read_pickle(cls, input_file):
    """Reads a format downstream data file."""
    with open(input_file, "rb") as f:
      data = pickle.load(f)
      return data


class LeaveProcessorBase(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir,rate,rng):
    """See base class."""
    path=os.path.join(data_dir,'train.pickle')
    return self._create_examples(path, "train",rate,rng)

  def get_dev_examples(self, data_dir,rate,rng):
    """See base class."""
    path = os.path.join(data_dir, 'eval.pickle')
    # path = os.path.join(data_dir, 'train.pickle')
    return self._create_examples(path, "dev",rate,rng)

  def get_test_examples(self, data_dir,rate,rng):
    """See base class."""
    path = os.path.join(data_dir, 'test.pickle')
    return self._create_examples(path, "test",rate,rng)

  def get_labels(self):
    """See base class."""
    return ["0", "1"]#0未离职，1离职

  def _create_examples(self, pickle_path, set_type,rate,rng):
    """Creates examples for the training and dev sets."""
    examples = []
    with open(pickle_path,'rb') as f:
        data=pickle.load(f)
    #example=[line[0],tokens_list,durings_list,positions_list,label]
    for (i, line) in enumerate(data):
      # Only the test set has a header
      guid=str(line[0][0])+'-'+str(line[0][1])
      tokens_list = line[1]
      durings_list = line[2]
      positions=line[3]
      label = line[4]
      if rate<=1:
          if rng.random() <= rate:
              examples.append(
                  MultiDaysInputExample(guid=guid, token_list=tokens_list, during_list=durings_list,
                                        positions=positions, label=label))
      else:
          for k in range(int(rate)):
              examples.append(
                  MultiDaysInputExample(guid=guid, token_list=tokens_list, during_list=durings_list,
                                        positions=positions, label=label))
          if rng.random() <= (rate-int(rate)):
              examples.append(
                  MultiDaysInputExample(guid=guid, token_list=tokens_list, during_list=durings_list,
                                        positions=positions, label=label))
      # examples.append(
      #     MultiDaysInputExample(guid=guid,token_list=tokens_list,during_list=durings_list,positions=positions,label=label))
    return examples

class PerfProcessorBase(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir,rate,rng):
    """See base class."""
    path=os.path.join(data_dir,'train.pickle')
    return self._create_examples(path, "train",rate,rng)

  def get_dev_examples(self, data_dir,rate,rng):
    """See base class."""
    path = os.path.join(data_dir, 'eval.pickle')
    # path = os.path.join(data_dir, 'train.pickle')
    return self._create_examples(path, "dev",rate,rng)

  def get_test_examples(self, data_dir,rate,rng):
    """See base class."""
    path = os.path.join(data_dir, 'test.pickle')
    return self._create_examples(path, "test",rate,rng)

  def get_labels(self):
    """See base class."""
    return ["0", "1"]#0高绩效，1低绩效

  def _create_examples(self, pickle_path, set_type,rate,rng):
    """Creates examples for the training and dev sets."""
    examples = []
    with open(pickle_path,'rb') as f:
        data=pickle.load(f)
    #example=[line[0],tokens_list,durings_list,positions_list,label]
    for (i, line) in enumerate(data):
      # Only the test set has a header
      guid=str(line[0][0])+'-'+str(line[0][1])
      tokens_list = line[1]
      durings_list = line[2]
      positions=line[3]
      label = line[4]
      if rate <= 1:
          if rng.random() <= rate:
              examples.append(
                  MultiDaysInputExample(guid=guid, token_list=tokens_list, during_list=durings_list,
                                        positions=positions, label=label))
      else:
          for k in range(int(rate)):
              examples.append(
                  MultiDaysInputExample(guid=guid, token_list=tokens_list, during_list=durings_list,
                                        positions=positions, label=label))
          if rng.random() <= (rate - int(rate)):
              examples.append(
                  MultiDaysInputExample(guid=guid, token_list=tokens_list, during_list=durings_list,
                                        positions=positions, label=label))

      # examples.append(
      #     MultiDaysInputExample(guid=guid,token_list=tokens_list,during_list=durings_list,positions=positions,label=label))
    return examples

class PerfProcessorBaseR(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir,rate,rng):
    """See base class."""
    path=os.path.join(data_dir,'train.pickle')
    return self._create_examples(path, "train",rate,rng)

  def get_dev_examples(self, data_dir,rate,rng):
    """See base class."""
    path = os.path.join(data_dir, 'eval.pickle')
    # path = os.path.join(data_dir, 'train.pickle')
    return self._create_examples(path, "dev",rate,rng)

  def get_test_examples(self, data_dir,rate,rng):
    """See base class."""
    path = os.path.join(data_dir, 'test.pickle')
    return self._create_examples(path, "test",rate,rng)

  def get_labels(self):
    """See base class."""
    return ["0", "1"]#0低绩效，1高绩效

  def _create_examples(self, pickle_path, set_type,rate,rng):
    """Creates examples for the training and dev sets."""
    examples = []
    with open(pickle_path,'rb') as f:
        data=pickle.load(f)
    #example=[line[0],tokens_list,durings_list,positions_list,label]
    for (i, line) in enumerate(data):
      # Only the test set has a header
      guid=str(line[0][0])+'-'+str(line[0][1])
      tokens_list = line[1]
      durings_list = line[2]
      positions=line[3]
      label = line[4]
      if label=='1':
          label='0'
      else:
          label='1'
      if rate <= 1:
          if rng.random() <= rate:
              examples.append(
                  MultiDaysInputExample(guid=guid, token_list=tokens_list, during_list=durings_list,
                                        positions=positions, label=label))
      else:
          for k in range(int(rate)):
              examples.append(
                  MultiDaysInputExample(guid=guid, token_list=tokens_list, during_list=durings_list,
                                        positions=positions, label=label))
          if rng.random() <= (rate - int(rate)):
              examples.append(
                  MultiDaysInputExample(guid=guid, token_list=tokens_list, during_list=durings_list,
                                        positions=positions, label=label))

      # examples.append(
      #     MultiDaysInputExample(guid=guid,token_list=tokens_list,during_list=durings_list,positions=positions,label=label))
    return examples

class Level3ProcessorBase(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir,rate,rng):
    """See base class."""
    path=os.path.join(data_dir,'train.pickle')
    return self._create_examples(path, "train",rate,rng)

  def get_dev_examples(self, data_dir,rate,rng):
    """See base class."""
    path = os.path.join(data_dir, 'eval.pickle')
    # path = os.path.join(data_dir, 'train.pickle')
    return self._create_examples(path, "dev",rate,rng)

  def get_test_examples(self, data_dir,rate,rng):
    """See base class."""
    path = os.path.join(data_dir, 'test.pickle')
    return self._create_examples(path, "test",rate,rng)

  def get_labels(self):
    """See base class."""
    return ["1", "2","3"]#1T,2P,3U,4AB,5M

  def _create_examples(self, pickle_path, set_type,rate,rng):
    """Creates examples for the training and dev sets."""
    examples = []
    with open(pickle_path,'rb') as f:
        data=pickle.load(f)
    #example=[line[0],tokens_list,durings_list,positions_list,label]
    for (i, line) in enumerate(data):
      # Only the test set has a header
      guid=str(line[0][0])+'-'+str(line[0][1])
      tokens_list = line[1]
      durings_list = line[2]
      positions=line[3]
      label = str(line[4])
      label_list=self.get_labels()
      if label in label_list:
          if rate <= 1:
              if rng.random() <= rate:
                  examples.append(
                      MultiDaysInputExample(guid=guid, token_list=tokens_list, during_list=durings_list,
                                            positions=positions, label=label))
          else:
              for k in range(int(rate)):
                  examples.append(
                      MultiDaysInputExample(guid=guid, token_list=tokens_list, during_list=durings_list,
                                            positions=positions, label=label))
              if rng.random() <= (rate - int(rate)):
                  examples.append(
                      MultiDaysInputExample(guid=guid, token_list=tokens_list, during_list=durings_list,
                                            positions=positions, label=label))
      # if random.random()<0.1:
      #       examples.append(
      #           MultiDaysInputExample(guid=guid,token_list=tokens_list,during_list=durings_list,positions=positions,label=label))
    return examples

class Level5ProcessorBase(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir,rate,rng):
    """See base class."""
    path=os.path.join(data_dir,'train.pickle')
    return self._create_examples(path, "train",rate,rng)

  def get_dev_examples(self, data_dir,rate,rng):
    """See base class."""
    path = os.path.join(data_dir, 'eval.pickle')
    # path = os.path.join(data_dir, 'train.pickle')
    return self._create_examples(path, "dev",rate,rng)

  def get_test_examples(self, data_dir,rate,rng):
    """See base class."""
    path = os.path.join(data_dir, 'test.pickle')
    return self._create_examples(path, "test",rate,rng)

  def get_labels(self):
    """See base class."""
    return ["1","2","3","4","5"]#1T,2P,3U,4AB,5M

  def _create_examples(self, pickle_path, set_type,rate,rng):
    """Creates examples for the training and dev sets."""
    examples = []
    with open(pickle_path,'rb') as f:
        data=pickle.load(f)
    #example=[line[0],tokens_list,durings_list,positions_list,label]
    for (i, line) in enumerate(data):
      # Only the test set has a header
      guid=str(line[0][0])+'-'+str(line[0][1])
      tokens_list = line[1]
      durings_list = line[2]
      positions=line[3]
      label = str(line[4])
      label_list=self.get_labels()
      if label in label_list:
          if rate <= 1:
              if rng.random() <= rate:
                  examples.append(
                      MultiDaysInputExample(guid=guid, token_list=tokens_list, during_list=durings_list,
                                            positions=positions, label=label))
          else:
              for k in range(int(rate)):
                  examples.append(
                      MultiDaysInputExample(guid=guid, token_list=tokens_list, during_list=durings_list,
                                            positions=positions, label=label))
              if rng.random() <= (rate - int(rate)):
                  examples.append(
                      MultiDaysInputExample(guid=guid, token_list=tokens_list, during_list=durings_list,
                                            positions=positions, label=label))
      # if random.random() < 0.1:
      #       examples.append(
      #           MultiDaysInputExample(guid=guid,token_list=tokens_list,during_list=durings_list,positions=positions,label=label))
    return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,max_day_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return MultiDaysInputFeatures(
        input_ids_list=[[0] * max_seq_length for i in range(max_day_length)],
        input_mask_list=[[0] * max_seq_length for i in range(max_day_length)],
        during_list=[[0] * max_seq_length for i in range(max_day_length)],
        positions=[0]*max_day_length,
        label_ids=0)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  token_list = example.token_list
  during_list=example.during_list
  positions=example.positions

  # if example.text_b:
  #   tokens_b = example.text_b

  # tokens_a = tokenizer.tokenize(example.text_a)
  # tokens_b = None
  # if example.text_b:
  #   tokens_b = tokenizer.tokenize(example.text_b)

  # if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  # else:
    # Account for [CLS] and [SEP] with "- 2"
    # if len(tokens_a) > max_seq_length - 2:
    #   tokens_a = tokens_a[0:(max_seq_length - 2)]

  if len(token_list) > max_day_length - 2:
      token_list = token_list[0:(max_day_length - 2)]
  for i,tokens in enumerate(token_list):
      if len(tokens) > max_seq_length - 2:
        token_list[i] = tokens[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  new_ids_list=[]
  new_mask_list=[]
  new_during_list=[]
  new_positions=[]
  for i,tokens in enumerate(token_list):
      new_tokens = []
      new_durings=[]
      new_tokens.append("[CLS]")
      new_durings.append(1)
      for j,token in enumerate(tokens):
          new_tokens.append(token)
          new_durings.append(during_list[i][j])
      new_tokens.append("[SEP]")
      new_durings.append(1)

      input_ids = tokenizer.convert_tokens_to_ids(new_tokens)

      input_mask = [1] * len(input_ids)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
          input_ids.append(0)
          input_mask.append(0)
          new_durings.append(0)


      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(new_durings) == max_seq_length
      new_ids_list.append(input_ids)
      new_mask_list.append(input_mask)
      new_during_list.append(new_durings)
      new_positions.append(positions[i])

  while len(new_ids_list) < max_day_length:
      new_ids_list.append([0]*max_seq_length)
      new_mask_list.append([0]*max_seq_length)
      new_during_list.append([0]*max_seq_length)
      new_positions.append(0)
  assert len(new_ids_list) == max_day_length
  assert len(new_mask_list) == max_day_length
  assert len(new_during_list) == max_day_length
  assert len(new_positions) == max_day_length


  # tokens = []
  # segment_ids = []
  # tokens.append("[CLS]")
  # segment_ids.append(0)
  # for token in tokens_a:
  #   tokens.append(token)
  #   segment_ids.append(0)
  # tokens.append("[SEP]")
  # segment_ids.append(0)
  #
  #
  # input_ids = tokenizer.convert_tokens_to_ids(tokens)
  #
  # # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # # tokens are attended to.
  # input_mask = [1] * len(input_ids)
  #
  # # Zero-pad up to the sequence length.
  # while len(input_ids) < max_seq_length:
  #   input_ids.append(0)
  #   input_mask.append(0)
  #   segment_ids.append(0)
  #
  # assert len(input_ids) == max_seq_length
  # assert len(input_mask) == max_seq_length
  # assert len(segment_ids) == max_seq_length

  label_ids = label_map[example.label]
  # print('-----------------')
  # print(tokens)

  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        # [tokenization.printable_text(x) for x in tokens]))
        [str(x) for x in token_list]))
    tf.logging.info("input_ids_list: %s" % " ".join([str(x) for x in new_ids_list]))
    tf.logging.info("input_mask_list: %s" % " ".join([str(x) for x in new_mask_list]))
    tf.logging.info("input_during_list: %s" % " ".join([str(x) for x in new_during_list]))
    tf.logging.info("input_positions: %s" % " ".join([str(x) for x in new_positions]))
    # tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_ids))

  feature = MultiDaysInputFeatures(
      input_ids_list=new_ids_list,
      input_mask_list=new_mask_list,
      during_list=new_during_list,
      positions=new_positions,
      label_ids=label_ids)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, max_day_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, max_day_length,tokenizer)


    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f
    def create_intlist_feature(values):
      list_values=[one for line in values for one in line]
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(list_values)))
      return f

    features = collections.OrderedDict()
    features["input_ids_list"] = create_intlist_feature(feature.input_ids_list)
    features["input_mask_list"] = create_intlist_feature(feature.input_mask_list)
    features["during_list"] = create_intlist_feature(feature.during_list)
    features["positions"] = create_int_feature(feature.positions)
    features["label_ids"] = create_int_feature([int(feature.label_ids)])
    features["list_shape"] = create_int_feature([-1,int(max_day_length),int(max_seq_length)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length,day_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids_list": tf.FixedLenFeature([day_length*seq_length], tf.int64),
      "input_mask_list": tf.FixedLenFeature([day_length*seq_length], tf.int64),
      "during_list": tf.FixedLenFeature([day_length*seq_length], tf.int64),
      "positions": tf.FixedLenFeature([day_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "list_shape":tf.FixedLenFeature([3], tf.int64)
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

get_shape_list=modeling.get_shape_list

def create_model(bert_config, is_training, input_ids_list, input_mask_list, during_list, position, labels, num_labels,sin_position):
        # bert_config, is_training, input_ids, input_mask, segment_ids,
        #          labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""


  input_shape=get_shape_list(input_ids_list,expected_rank=3)
  batch_size = input_shape[0]
  day_length = input_shape[1]
  route_length=input_shape[2]
  input_ids=tf.reshape(input_ids_list,[batch_size*day_length,route_length])
  segment_during=tf.reshape(during_list,[batch_size*day_length,route_length])
  input_mask=tf.reshape(input_mask_list,[batch_size*day_length,route_length])

  # model = modeling.RertModel(
  #     config=bert_config,
  #     is_training=is_training,
  #     input_ids_list=input_ids_list,
  #     input_mask_list=input_mask_list,
  #     input_during_list=during_list,
  #     position=position)
  model = modeling.RertModel(
    config=bert_config,
    is_training=is_training,
    input_ids=input_ids,
    input_during=segment_during,
    input_mask=input_mask,
    sin_position=sin_position)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value
  embedding_table=model.get_embedding_table()

  days_input = tf.reshape(output_layer,[batch_size,day_length,hidden_size])
  days_mask=tf.reduce_max(input_mask_list,axis=2)
  CLS = embedding_table[101]  # same with Rert
  # CLS = embedding_table[94]  # defferent with Rert
  CLS = tf.expand_dims(CLS, axis=0)
  CLS = tf.tile(CLS, [batch_size, 1])

  # CLS = tf.reduce_mean(days_input,axis=1)  # averange of Rert output

  CLS=tf.expand_dims(CLS,axis=1)

  days_input=tf.concat([CLS,days_input],axis=1)
  CLS_one=tf.tile(tf.constant([[1]]),[batch_size,1])
  print(CLS_one)
  print(days_mask)
  print(position)
  # exit()
  days_mask=tf.concat([CLS_one,days_mask],axis=1)
  position=tf.concat([CLS_one,position],axis=1)
  Dmodel=modeling.DRertModel_new(
      config=bert_config,
      is_training=is_training,
      input_embeddings=days_input,
      input_during=position,
      input_mask=days_mask,
      sin_position=sin_position)
  output_layer = Dmodel.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings,sin_position,stop_gradient,gradient_accumulation_multiplier,binary_threshold):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids_list=features["input_ids_list"]
    input_mask_list=features["input_mask_list"]
    during_list=features["during_list"]
    position=features["positions"]
    label_ids=features["label_ids"]
    list_shape=features['list_shape']

    print(input_ids_list)
    print(input_mask_list)
    print(during_list)
    print(position)
    print(label_ids)
    print(list_shape)

    input_ids_list=tf.reshape(input_ids_list,list_shape[0])
    input_mask_list=tf.reshape(input_mask_list,list_shape[0])
    during_list=tf.reshape(during_list,list_shape[0])
    # input_ids = features["input_ids"]
    # input_mask = features["input_mask"]
    # segment_ids = features["segment_ids"]
    # label_ids = features["label_ids"]
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids_list, input_mask_list, during_list, position, label_ids,
        num_labels,sin_position)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_ag_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu,gradient_accumulation_multiplier,stop_gradient,initialized_variable_names)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example,probabilities):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        if num_labels==2:
            if binary_threshold>0:
                balance0=0.5/(1-binary_threshold)
                balance1=0.5/(binary_threshold)
                threshold_balance=tf.constant([[balance0,0],[0,balance1]],dtype=tf.float32)
                threshold_pro=tf.matmul(probabilities,threshold_balance)
                predictions = tf.argmax(threshold_pro, axis=-1, output_type=tf.int32)

        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)

        precision = tf_metrics.precision(
            # label_ids, predictions, num_labels, average='micro')
            label_ids, predictions, num_labels, average='weighted')
        recall = tf_metrics.recall(
            # label_ids, predictions, num_labels, average='micro')
            label_ids, predictions, num_labels, average='weighted')
        micro_f1 = tf_metrics.f1(
            label_ids, predictions, num_labels, average='micro')
        macro_f1 = tf_metrics.f1(
            label_ids, predictions, num_labels, average='macro')
        weighted_f1 = tf_metrics.f1(
            label_ids, predictions, num_labels, average='weighted')
        cm = tf_metrics.confusion_matrix(
            label_ids, predictions, num_labels)
        if num_labels==2:
            precision = tf_metrics.precision(
                label_ids, predictions, num_labels, pos_indices=[1])
            recall = tf_metrics.recall(
                label_ids, predictions, num_labels, pos_indices=[1])
            precision2 =tf.metrics.precision(label_ids, predictions)
            recall2 =tf.metrics.recall(label_ids, predictions)
            positive_pro = tf.squeeze(probabilities[:, 1])
            roc_auc=tf.metrics.auc(label_ids,positive_pro,curve='ROC')
            pr_auc=tf.metrics.auc(label_ids,positive_pro,curve='PR')

        # recall=tf.metrics.recall(label_ids,predictions)
        # precision=tf.metrics.precision(label_ids,predictions,)
        # auc=tf.metrics.auc(label_ids,predictions)

        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        if num_labels==2:
            return {
                "eval_accuracy": accuracy,
                "eval_loss": loss,
                "precision": precision,
                "recall": recall,
                "precision2": precision2,
                "recall2": recall2,
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "confusion_matrix": cm,
                "roc_auc":roc_auc,
                "pr_auc":pr_auc
            }
        else:
            return {
                "eval_accuracy": accuracy,
                "eval_loss": loss,
                "precision":precision,
                "recall":recall,
                "micro_f1":micro_f1,
                "macro_f1":macro_f1,
                "weighted_f1":weighted_f1,
                "confusion_matrix":cm
            }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example,probabilities])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_device

  processors = {
      # 'perfbase': ,
      # 'perfbase': LeaveProcessorBase,
      'leavebase': LeaveProcessorBase,
      'perfbase': PerfProcessorBase,
      'perfbaser': PerfProcessorBaseR,
      'level3base': Level3ProcessorBase,
      'level5base': Level5ProcessorBase,
      # 'nextbase': NextProcessorBase,
      # 'schedulebase': ScheduleProcessorBase,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  # XLA additional config
  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = True
  optimizer_options = session_config.graph_options.optimizer_options
  optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  rng = random.Random(FLAGS.random_seed)
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir,rate=FLAGS.data_sampling_rate,rng=rng)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  binary_threshold=FLAGS.binary_threshold
  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      # num_train_steps=FLAGS.num_train_steps,
      num_train_steps=num_train_steps,
      # num_warmup_steps=FLAGS.num_warmup_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      sin_position=FLAGS.sin_position,
      stop_gradient=FLAGS.stop_gradient,
      gradient_accumulation_multiplier=FLAGS.gradient_accumulation_multiplier,
      binary_threshold=binary_threshold)

  # model_fn = model_fn_builder(
  #     bert_config=bert_config,
  #     num_labels=len(label_list),
  #     init_checkpoint=FLAGS.init_checkpoint,
  #     learning_rate=FLAGS.learning_rate,
  #     num_train_steps=num_train_steps,
  #     num_warmup_steps=num_warmup_steps,
  #     use_tpu=FLAGS.use_tpu,
  #     use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length,FLAGS.max_day_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        day_length=FLAGS.max_day_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  do_eval_train_set=True
  if do_eval_train_set:

    eval_file = os.path.join(FLAGS.output_dir, "train.tf_record")

    tf.logging.info("***** Running evaluation train set *****")
    # This tells the estimator to run through the entire set.
    eval_steps = None

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        day_length=FLAGS.max_day_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_train_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval_TrainSet results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir,rate=FLAGS.data_sampling_rate,rng=rng)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(PaddingInputExample())

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length,FLAGS.max_day_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    leav_num=0
    for one in eval_examples:
        if one.label=='1':
            leav_num+=1
    print('leav_num:{}'.format(leav_num))
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        day_length=FLAGS.max_day_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir,rate=FLAGS.data_sampling_rate,rng=rng)
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length,FLAGS.max_day_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        day_length=FLAGS.max_day_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
