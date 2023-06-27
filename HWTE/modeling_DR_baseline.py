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
"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import numpy as np
import six
import tensorflow as tf
import position_embedding as pe
import performer_fast_attention_pro as fast_attention


class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


class LongformerModel(object):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    # global_mask=tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
    global_mask_value=[[0 for j in range(seq_length)] for i in range(batch_size)]
    for i in range(batch_size):
        global_mask_value[i][0]=1
    global_mask=tf.constant(value=global_mask_value, dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.

        # attention_mask = create_attention_mask_from_input_mask(
        #     input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        # self.all_encoder_layers = transformer_model(
        #     input_tensor=self.embedding_output,
        #     attention_mask=attention_mask,
        #     hidden_size=config.hidden_size,
        #     num_hidden_layers=config.num_hidden_layers,
        #     num_attention_heads=config.num_attention_heads,
        #     intermediate_size=config.intermediate_size,
        #     intermediate_act_fn=get_activation(config.hidden_act),
        #     hidden_dropout_prob=config.hidden_dropout_prob,
        #     attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        #     initializer_range=config.initializer_range,
        #     do_return_all_layers=True)

        self.all_encoder_layers = longformer_model(
            input_tensor=self.embedding_output,
            input_mask=input_mask,
            global_mask=global_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True,
            is_training=is_training)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table


class LongformerGoModel(object):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None,
               ):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    # global_mask=tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    # global_mask_value=[[0 for j in range(seq_length)] for i in range(batch_size)]
    # for i in range(batch_size):
    #     global_mask_value[i][0]=1
    # global_mask=tf.constant(value=global_mask_value, dtype=tf.int32)

    # global_mask_value=[[0 for j in range(8192)] for i in range(2)]
    # for i in range(2):
    #     global_mask_value[i][0]=1
    # global_mask=tf.constant(value=global_mask_value, dtype=tf.int32)


    global_mask_b=tf.zeros([batch_size,seq_length-1],dtype=tf.int32)
    global_mask_a=tf.ones([batch_size,1],dtype=tf.int32)
    global_mask=tf.concat([global_mask_a,global_mask_b],axis=1)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.

        # attention_mask = create_attention_mask_from_input_mask(
        #     input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        # self.all_encoder_layers = transformer_model(
        #     input_tensor=self.embedding_output,
        #     attention_mask=attention_mask,
        #     hidden_size=config.hidden_size,
        #     num_hidden_layers=config.num_hidden_layers,
        #     num_attention_heads=config.num_attention_heads,
        #     intermediate_size=config.intermediate_size,
        #     intermediate_act_fn=get_activation(config.hidden_act),
        #     hidden_dropout_prob=config.hidden_dropout_prob,
        #     attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        #     initializer_range=config.initializer_range,
        #     do_return_all_layers=True)

        self.all_encoder_layers = longformer_go_model(
            input_tensor=self.embedding_output,
            input_mask=input_mask,
            global_mask=global_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True,
            is_training=is_training)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table


class LongformerSildModel(object):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    # global_mask=tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
    # global_mask_value=[[0 for j in range(seq_length)] for i in range(batch_size)]
    # for i in range(batch_size):
    #     global_mask_value[i][0]=1
    # global_mask=tf.constant(value=global_mask_value, dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.

        # attention_mask = create_attention_mask_from_input_mask(
        #     input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        # self.all_encoder_layers = transformer_model(
        #     input_tensor=self.embedding_output,
        #     attention_mask=attention_mask,
        #     hidden_size=config.hidden_size,
        #     num_hidden_layers=config.num_hidden_layers,
        #     num_attention_heads=config.num_attention_heads,
        #     intermediate_size=config.intermediate_size,
        #     intermediate_act_fn=get_activation(config.hidden_act),
        #     hidden_dropout_prob=config.hidden_dropout_prob,
        #     attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        #     initializer_range=config.initializer_range,
        #     do_return_all_layers=True)

        self.all_encoder_layers = longformer_slide_model(
            input_tensor=self.embedding_output,
            input_mask=input_mask,
            # global_mask=global_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True,
            is_training=is_training)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table

class LSTMModel(object):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    # global_mask=tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
    # global_mask_value=[[0 for j in range(seq_length)] for i in range(batch_size)]
    # for i in range(batch_size):
    #     global_mask_value[i][0]=1
    # global_mask=tf.constant(value=global_mask_value, dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.

        # self.embedding_output = embedding_postprocessor(
        #     input_tensor=self.embedding_output,
        #     use_token_type=True,
        #     token_type_ids=token_type_ids,
        #     token_type_vocab_size=config.type_vocab_size,
        #     token_type_embedding_name="token_type_embeddings",
        #     use_position_embeddings=True,
        #     position_embedding_name="position_embeddings",
        #     initializer_range=config.initializer_range,
        #     max_position_embeddings=config.max_position_embeddings,
        #     dropout_prob=config.hidden_dropout_prob)

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.

        # attention_mask = create_attention_mask_from_input_mask(
        #     input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        # self.all_encoder_layers = transformer_model(
        #     input_tensor=self.embedding_output,
        #     attention_mask=attention_mask,
        #     hidden_size=config.hidden_size,
        #     num_hidden_layers=config.num_hidden_layers,
        #     num_attention_heads=config.num_attention_heads,
        #     intermediate_size=config.intermediate_size,
        #     intermediate_act_fn=get_activation(config.hidden_act),
        #     hidden_dropout_prob=config.hidden_dropout_prob,
        #     attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        #     initializer_range=config.initializer_range,
        #     do_return_all_layers=True)

        self.all_encoder_layers = lstm_model(
            input_tensor=self.embedding_output,
            input_mask=input_mask,
            # global_mask=global_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True,
            is_training=is_training)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table

class PerformerModel(object):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    # global_mask=tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
    # global_mask_value=[[0 for j in range(seq_length)] for i in range(batch_size)]
    # for i in range(batch_size):
    #     global_mask_value[i][0]=1
    # global_mask=tf.constant(value=global_mask_value, dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.

        # attention_mask = create_attention_mask_from_input_mask(
        #     input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        # self.all_encoder_layers = transformer_model(
        #     input_tensor=self.embedding_output,
        #     attention_mask=attention_mask,
        #     hidden_size=config.hidden_size,
        #     num_hidden_layers=config.num_hidden_layers,
        #     num_attention_heads=config.num_attention_heads,
        #     intermediate_size=config.intermediate_size,
        #     intermediate_act_fn=get_activation(config.hidden_act),
        #     hidden_dropout_prob=config.hidden_dropout_prob,
        #     attention_probs_dropout_prob=config.attention_probs_dropout_prob,
        #     initializer_range=config.initializer_range,
        #     do_return_all_layers=True)

        self.all_encoder_layers = performer_model(
            input_tensor=self.embedding_output,
            input_mask=input_mask,
            # global_mask=global_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True,
            is_training=is_training)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table

class BertModel(object):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table

class RertModel(object):
  """RERT model ("Routing Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_during=None,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               sin_position=False,
               scope=None):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="rert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        # self.embedding_output = embedding_postprocessor(
        #     input_tensor=self.embedding_output,
        #     use_token_type=True,
        #     token_type_ids=token_type_ids,
        #     token_type_vocab_size=config.type_vocab_size,
        #     token_type_embedding_name="token_type_embeddings",
        #     use_position_embeddings=True,
        #     position_embedding_name="position_embeddings",
        #     initializer_range=config.initializer_range,
        #     max_position_embeddings=config.max_position_embeddings,
        #     dropout_prob=config.hidden_dropout_prob)

        max_position_embeddings_during = config.max_position_embeddings
        start_during_table = [[0 if j < i else 1 for j in range(max_position_embeddings_during)] for i in
                              range(max_position_embeddings_during)]
        end_during_table = [[1 if j < i else 0 for j in range(max_position_embeddings_during)] for i in
                            range(max_position_embeddings_during)]
        end_during_table=tf.constant(end_during_table,dtype=tf.int32)
        start_during_table=tf.constant(start_during_table,dtype=tf.int32)

        self.embedding_output,self.position_embedding = embedding_postprocessor_during(
            input_tensor=self.embedding_output,
            use_during_embedding=True,
            token_during=input_during,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            sin_position=sin_position,
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob,
            start_during_table=start_during_table,
            end_during_table=end_during_table)

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_position_embedding(self):
      return self.position_embedding

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table

class DRertModel(object):
  """RERT model ("Routing Encoder Representations from Transformers").

  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               input_embeddings,
               input_during=None,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               sin_position=False,
               scope=None):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    # input_shape = get_shape_list(input_ids, expected_rank=2)
    input_shape = get_shape_list(input_embeddings, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="drert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        self.embedding_output=input_embeddings
        # (self.embedding_output, self.embedding_table) = embedding_lookup(
        #     input_ids=input_ids,
        #     vocab_size=config.vocab_size,
        #     embedding_size=config.hidden_size,
        #     initializer_range=config.initializer_range,
        #     word_embedding_name="word_embeddings",
        #     use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        # self.embedding_output = embedding_postprocessor(
        #     input_tensor=self.embedding_output,
        #     use_token_type=True,
        #     token_type_ids=token_type_ids,
        #     token_type_vocab_size=config.type_vocab_size,
        #     token_type_embedding_name="token_type_embeddings",
        #     use_position_embeddings=True,
        #     position_embedding_name="position_embeddings",
        #     initializer_range=config.initializer_range,
        #     max_position_embeddings=config.max_position_embeddings,
        #     dropout_prob=config.hidden_dropout_prob)

        max_position_embeddings_during = config.max_position_embeddings
        start_during_table = [[0 if j < i else 1 for j in range(max_position_embeddings_during)] for i in
                              range(max_position_embeddings_during)]
        end_during_table = [[1 if j < i else 0 for j in range(max_position_embeddings_during)] for i in
                            range(max_position_embeddings_during)]
        end_during_table=tf.constant(end_during_table,dtype=tf.int32)
        start_during_table=tf.constant(start_during_table,dtype=tf.int32)

        self.embedding_output,self.position_embedding = embedding_postprocessor_during(
            input_tensor=self.embedding_output,
            use_during_embedding=True,
            token_during=input_during,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            sin_position=sin_position,
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob,
            start_during_table=start_during_table,
            end_during_table=end_during_table)

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = create_attention_mask_from_input_mask(
            input_embeddings, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_position_embedding(self):
      return self.position_embedding

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table

class DRertModel_new(object):
  """RERT model ("Routing Encoder Representations from Transformers").
  input_during is not during but abs position
  Example usage:

  ```python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  ```
  """

  def __init__(self,
               config,
               is_training,
               input_embeddings,
               input_during=None,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               sin_position=False,
               scope=None):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    # input_shape = get_shape_list(input_ids, expected_rank=2)
    input_shape = get_shape_list(input_embeddings, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="drert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        self.embedding_output=input_embeddings
        # (self.embedding_output, self.embedding_table) = embedding_lookup(
        #     input_ids=input_ids,
        #     vocab_size=config.vocab_size,
        #     embedding_size=config.hidden_size,
        #     initializer_range=config.initializer_range,
        #     word_embedding_name="word_embeddings",
        #     use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        # self.embedding_output = embedding_postprocessor(
        #     input_tensor=self.embedding_output,
        #     use_token_type=True,
        #     token_type_ids=token_type_ids,
        #     token_type_vocab_size=config.type_vocab_size,
        #     token_type_embedding_name="token_type_embeddings",
        #     use_position_embeddings=True,
        #     position_embedding_name="position_embeddings",
        #     initializer_range=config.initializer_range,
        #     max_position_embeddings=config.max_position_embeddings,
        #     dropout_prob=config.hidden_dropout_prob)

        max_position_embeddings_during = config.max_position_embeddings
        start_during_table = [[0 if j < i else 1 for j in range(max_position_embeddings_during)] for i in
                              range(max_position_embeddings_during)]
        end_during_table = [[1 if j < i else 0 for j in range(max_position_embeddings_during)] for i in
                            range(max_position_embeddings_during)]
        end_during_table=tf.constant(end_during_table,dtype=tf.int32)
        start_during_table=tf.constant(start_during_table,dtype=tf.int32)

        self.embedding_output,self.position_embedding = embedding_postprocessor_abs_position(
            input_tensor=self.embedding_output,
            use_during_embedding=True,
            token_position=input_during,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            sin_position=sin_position,
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob,
            start_during_table=start_during_table,
            end_during_table=end_during_table)

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        attention_mask = create_attention_mask_from_input_mask(
            input_embeddings, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_position_embedding(self):
      return self.position_embedding

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def dropout(input_tensor, dropout_prob):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  return output

def dropout_TF(input_tensor, dropout_prob,is_training=True):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if dropout_prob is None or dropout_prob == 0.0:
    return input_tensor
  if is_training:
    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output
  else:
    return input_tensor


def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
  """Runs layer normalization followed by dropout."""
  output_tensor = layer_norm(input_tensor, name)
  output_tensor = dropout(output_tensor, dropout_prob)
  return output_tensor


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)


def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
  """Looks up words embeddings for id tensor.

  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
      ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialization range.
    word_embedding_name: string. Name of the embedding table.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.gather()`.

  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))

  flat_input_ids = tf.reshape(input_ids, [-1])
  if use_one_hot_embeddings:
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    output = tf.gather(embedding_table, flat_input_ids)

  input_shape = get_shape_list(input_ids)

  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return (output, embedding_table)


def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

  if use_position_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      full_position_embeddings=pe.sinusoidal_positional_embedding(max_position_embeddings, width)
      # full_position_embeddings = tf.get_variable(
      #     name=position_embedding_name,
      #     shape=[max_position_embeddings, width],
      #     initializer=create_initializer(initializer_range))

      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

  output = layer_norm_and_dropout(output, dropout_prob)
  return output

# max_position_embeddings_during=2048
max_position_embeddings_during=512
start_during_table=[[0 if j<i else 1 for j in range(max_position_embeddings_during)] for i in range(max_position_embeddings_during)]
end_during_table=[[1 if j<i else 0 for j in range(max_position_embeddings_during)] for i in range(max_position_embeddings_during)]
# end_during_table=tf.constant(end_during_table,dtype=tf.int32)
# start_during_table=tf.constant(start_during_table,dtype=tf.int32)

def positional_embedding(pos_seq, inv_freq, bsz=None):
    sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    if bsz is not None:
        return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
    else:
        return pos_emb[:, None, :]

def embedding_postprocessor_during(input_tensor,
                            use_during_embedding=True,
                            token_during=None,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            sin_position=False,
                            initializer_range=0.02,
                            max_position_embeddings=2048,
                            dropout_prob=0.1,
                            start_during_table=None,
                            end_during_table=None
                            ):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_during_embedding: bool. Whether to add embeddings for `token_during`.
    token_during: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_during_embedding` is True.
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_during_embedding:
      if token_during is None:
          raise ValueError("`token_during` must be specified if"
                           "`use_during_embedding` is True.")
      token_during_divisor=tf.expand_dims(token_during, axis=-1)
      end_ids=tf.cumsum(token_during,axis=-1)
      start_ids=tf.cumsum(token_during,axis=-1,exclusive=True)
      start_table = tf.nn.embedding_lookup(start_during_table, start_ids)
      end_table = tf.nn.embedding_lookup(end_during_table, end_ids)
      during_table = start_table * end_table
      mean_during_table=tf.to_float(during_table)/(tf.to_float(token_during_divisor)+1e-5)
      mean_during_table=tf.to_float(mean_during_table)

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

  if use_position_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      if sin_position:
          pos_seq = tf.range(max_position_embeddings - 1, -1, -1.0)
          inv_freq = [1 / (10000.0 ** (i / width)) for i in range(0, width, 2)]
          inv_freq = tf.constant(inv_freq)
          full_position_embeddings=positional_embedding(pos_seq, inv_freq)
          full_position_embeddings=tf.squeeze(full_position_embeddings)
          # print(full_position_embeddings)
          # exit()
      else:
          full_position_embeddings = tf.get_variable(
              name=position_embedding_name,
              shape=[max_position_embeddings, width],
              initializer=create_initializer(initializer_range))
          # Since the position embedding table is a learned variable, we create it
          # using a (long) sequence length `max_position_embeddings`. The actual
          # sequence length might be shorter than this, for faster training of
          # tasks that do not have long sequences.
          #
          # So `full_position_embeddings` is effectively an embedding table
          # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
          # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
          # perform a slice.
      if use_during_embedding:
          during_and_position_embeddings=tf.einsum('blt,td->bld',mean_during_table,full_position_embeddings)
          output += during_and_position_embeddings
      else:
          position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                         [seq_length, -1])
          num_dims = len(output.shape.as_list())

          # Only the last two dimensions are relevant (`seq_length` and `width`), so
          # we broadcast among the first dimensions, which is typically just
          # the batch size.
          position_broadcast_shape = []
          for _ in range(num_dims - 2):
            position_broadcast_shape.append(1)
          position_broadcast_shape.extend([seq_length, width])
          position_embeddings = tf.reshape(position_embeddings,
                                           position_broadcast_shape)
          output += position_embeddings

  output = layer_norm_and_dropout(output, dropout_prob)
  if use_position_embeddings:
    if use_during_embedding:
        return output, during_and_position_embeddings
    else:
        return output,position_embeddings
  else:
    return output,output

def embedding_postprocessor_abs_position(input_tensor,
                            use_during_embedding=True,
                            token_position=None,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            sin_position=False,
                            initializer_range=0.02,
                            max_position_embeddings=2048,
                            dropout_prob=0.1,
                            start_during_table=None,
                            end_during_table=None
                            ):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_during_embedding: bool. Whether to add embeddings for `token_during`.
    token_during: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_during_embedding` is True.
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  token_during=token_position

  if use_during_embedding:
      if token_during is None:
          raise ValueError("`token_during` must be specified if"
                           "`use_during_embedding` is True.")
      token_during_divisor=tf.expand_dims(token_during, axis=-1)
      # end_ids=tf.cumsum(token_during,axis=-1)
      # start_ids=tf.cumsum(token_during,axis=-1,exclusive=True)
      bais=tf.constant(1,dtype=tf.int32)
      end_ids=tf.add(token_during,bais)
      start_ids=token_during
      start_table = tf.nn.embedding_lookup(start_during_table, start_ids)
      end_table = tf.nn.embedding_lookup(end_during_table, end_ids)
      during_table = start_table * end_table
      mean_during_table=tf.to_float(during_table)/(tf.to_float(token_during_divisor)+1e-5)
      mean_during_table=tf.to_float(mean_during_table)

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

  if use_position_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      if sin_position:
          pos_seq = tf.range(max_position_embeddings - 1, -1, -1.0)
          inv_freq = [1 / (10000.0 ** (i / width)) for i in range(0, width, 2)]
          inv_freq = tf.constant(inv_freq)
          full_position_embeddings=positional_embedding(pos_seq, inv_freq)
          full_position_embeddings=tf.squeeze(full_position_embeddings)
          # print(full_position_embeddings)
          # exit()
      else:
          full_position_embeddings = tf.get_variable(
              name=position_embedding_name,
              shape=[max_position_embeddings, width],
              initializer=create_initializer(initializer_range))
          # Since the position embedding table is a learned variable, we create it
          # using a (long) sequence length `max_position_embeddings`. The actual
          # sequence length might be shorter than this, for faster training of
          # tasks that do not have long sequences.
          #
          # So `full_position_embeddings` is effectively an embedding table
          # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
          # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
          # perform a slice.
      if use_during_embedding:
          during_and_position_embeddings=tf.einsum('blt,td->bld',mean_during_table,full_position_embeddings)
          output += during_and_position_embeddings
      else:
          position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                         [seq_length, -1])
          position_embeddings
          num_dims = len(output.shape.as_list())

          # Only the last two dimensions are relevant (`seq_length` and `width`), so
          # we broadcast among the first dimensions, which is typically just
          # the batch size.
          position_broadcast_shape = []
          for _ in range(num_dims - 2):
            position_broadcast_shape.append(1)
          position_broadcast_shape.extend([seq_length, width])
          position_embeddings = tf.reshape(position_embeddings,
                                           position_broadcast_shape)
          output += position_embeddings

  output = layer_norm_and_dropout(output, dropout_prob)
  if use_position_embeddings:
    if use_during_embedding:
        return output, during_and_position_embeddings
    else:
        return output,position_embeddings
  else:
    return output,output


def create_attention_mask_from_input_mask(from_tensor, to_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    float Tensor of shape [batch_size, from_seq_length, to_seq_length].
  """
  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  batch_size = from_shape[0]
  from_seq_length = from_shape[1]

  to_shape = get_shape_list(to_mask, expected_rank=2)
  to_seq_length = to_shape[1]

  to_mask = tf.cast(
      tf.reshape(to_mask, [batch_size, 1, to_seq_length]), tf.float32)

  # We don't assume that `from_tensor` is a mask (although it could be). We
  # don't actually care if we attend *from* padding tokens (only *to* padding)
  # tokens so we create a tensor of all ones.
  #
  # `broadcast_ones` = [batch_size, from_seq_length, 1]
  broadcast_ones = tf.ones(
      shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

  # Here we broadcast along two dimensions to create the mask.
  mask = broadcast_ones * to_mask

  return mask


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head.
    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].
    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.

  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`

  from_tensor_2d = reshape_to_matrix(from_tensor)
  to_tensor_2d = reshape_to_matrix(to_tensor)

  # `query_layer` = [B*F, N*H]
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      kernel_initializer=create_initializer(initializer_range))

  # `key_layer` = [B*T, N*H]
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      kernel_initializer=create_initializer(initializer_range))

  # `query_layer` = [B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  return context_layer


# shape_list = get_shape_list
def shape_list(tensor: tf.Tensor):
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (:obj:`tf.Tensor`): The tensor we want the shape of.

    Returns:
        :obj:`List[int]`: The shape of the tensor as a list.
    """
    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class LongformerAttentionModule():
    def __init__(self,attention_window,layer_id):
        assert (
                attention_window % 2 == 0
        ), f"`attention_window` for layer has to be an even value. Given {attention_window}"
        assert (
                attention_window > 0
        ), f"`attention_window` for layer has to be positive. Given {attention_window}"

        self.attention_window=attention_window
        self.one_sided_attn_window_size = attention_window // 2
        self.layer_id=layer_id

    def _sliding_chunks_query_key_matmul(self, query, key, window_overlap):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer) with an
        overlap of size window_overlap
        """
        batch_size, seq_len, num_heads, head_dim = shape_list(query)

        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                seq_len % (window_overlap * 2),
                0,
                message=f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}",
            )
            tf.debugging.assert_equal(
                shape_list(query),
                shape_list(key),
                message=f"Shape of query and key should be equal, but got query: {shape_list(query)} and key: {shape_list(key)}",
            )

        chunks_count = seq_len // window_overlap - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = tf.reshape(
            tf.transpose(query, (0, 2, 1, 3)),
            (batch_size * num_heads, seq_len, head_dim),
        )
        key = tf.reshape(tf.transpose(key, (0, 2, 1, 3)), (batch_size * num_heads, seq_len, head_dim))
        chunked_query = self._chunk(query, window_overlap)
        chunked_key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        chunked_query = tf.cast(chunked_query, dtype=chunked_key.dtype)
        chunked_attention_scores = tf.einsum("bcxd,bcyd->bcxy", chunked_query, chunked_key)  # multiply

        # convert diagonals into columns
        paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 1], [0, 0]])
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(chunked_attention_scores, paddings)

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        # TODO: This code is most likely not very efficient and should be improved
        diagonal_attn_scores_up_triang = tf.concat(
            [
                diagonal_chunked_attention_scores[:, :, :window_overlap, : window_overlap + 1],
                diagonal_chunked_attention_scores[:, -1:, window_overlap:, : window_overlap + 1],
            ],
            axis=1,
        )

        # - copying the lower triangle
        diagonal_attn_scores_low_triang = tf.concat(
            [
                tf.zeros(
                    (batch_size * num_heads, 1, window_overlap, window_overlap),
                    dtype=diagonal_chunked_attention_scores.dtype,
                ),
                diagonal_chunked_attention_scores[:, :, -(window_overlap + 1): -1, window_overlap + 1:],
            ],
            axis=1,
        )
        diagonal_attn_scores_first_chunk = tf.concat(
            [
                tf.roll(
                    diagonal_chunked_attention_scores,
                    shift=[1, window_overlap],
                    axis=[2, 3],
                )[:, :, :window_overlap, :window_overlap],
                tf.zeros(
                    (batch_size * num_heads, 1, window_overlap, window_overlap),
                    dtype=diagonal_chunked_attention_scores.dtype,
                ),
            ],
            axis=1,
        )
        first_chunk_mask = (
                tf.tile(
                    tf.range(chunks_count + 1)[None, :, None, None],
                    (batch_size * num_heads, 1, window_overlap, window_overlap),
                )
                < 1
        )
        diagonal_attn_scores_low_triang = tf.where(
            first_chunk_mask,
            diagonal_attn_scores_first_chunk,
            diagonal_attn_scores_low_triang,
        )

        # merging upper and lower triangle
        diagonal_attention_scores = tf.concat(
            [diagonal_attn_scores_low_triang, diagonal_attn_scores_up_triang], axis=-1
        )

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = tf.transpose(
            tf.reshape(
                diagonal_attention_scores,
                (batch_size, num_heads, seq_len, 2 * window_overlap + 1),
            ),
            (0, 2, 1, 3),
        )

        diagonal_attention_scores = self._mask_invalid_locations(diagonal_attention_scores, window_overlap)

        return diagonal_attention_scores

    @staticmethod
    def _mask_invalid_locations(input_tensor, window_overlap):
        # create correct upper triangle bool mask
        mask_2d_upper = tf.reverse(
            tf.linalg.band_part(tf.ones(shape=(window_overlap, window_overlap + 1)), -1, 0),
            axis=[0],
        )

        # pad to full matrix
        padding = tf.convert_to_tensor(
            [[0, shape_list(input_tensor)[1] - window_overlap], [0, shape_list(input_tensor)[3] - window_overlap - 1]]
        )

        # create lower mask
        mask_2d = tf.pad(mask_2d_upper, padding)

        # combine with upper mask
        mask_2d = mask_2d + tf.reverse(mask_2d, axis=[0, 1])

        # broadcast to full matrix
        # mask_4d = tf.tile(mask_2d[None, :, None, :], (shape_list(input_tensor)[0], 1, 1, 1))
        mask_4d = tf.tile(mask_2d[None, :, None, :], (shape_list(input_tensor)[0], 1, shape_list(input_tensor)[2], 1))

        # inf tensor used for masking
        inf_tensor = -float("inf") * tf.ones_like(input_tensor)

        # mask
        input_tensor = tf.where(tf.math.greater(mask_4d, 0), inf_tensor, input_tensor)

        return input_tensor

    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs, value, window_overlap):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """

        batch_size, seq_len, num_heads, head_dim = shape_list(value)

        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                seq_len % (window_overlap * 2),
                0,
                message="Seq_len has to be multiple of 2 * window_overlap",
            )
            tf.debugging.assert_equal(
                shape_list(attn_probs)[:3],
                shape_list(value)[:3],
                message="value and attn_probs must have same dims (except head_dim)",
            )
            tf.debugging.assert_equal(
                shape_list(attn_probs)[3],
                2 * window_overlap + 1,
                message="attn_probs last dim has to be 2 * window_overlap + 1",
            )

        chunks_count = seq_len // window_overlap - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap
        chunked_attn_probs = tf.reshape(
            tf.transpose(attn_probs, (0, 2, 1, 3)),
            (
                batch_size * num_heads,
                seq_len // window_overlap,
                window_overlap,
                2 * window_overlap + 1,
            ),
        )

        # group batch_size and num_heads dimensions into one
        value = tf.reshape(
            tf.transpose(value, (0, 2, 1, 3)),
            (batch_size * num_heads, seq_len, head_dim),
        )

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        paddings = tf.convert_to_tensor([[0, 0], [window_overlap, window_overlap], [0, 0]])
        padded_value = tf.pad(value, paddings, constant_values=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        frame_size = 3 * window_overlap * head_dim
        frame_hop_size = (shape_list(padded_value)[1] * head_dim - frame_size) // chunks_count
        chunked_value = tf.signal.frame(
            tf.reshape(padded_value, (batch_size * num_heads, -1)),
            frame_size,
            frame_hop_size,
        )
        chunked_value = tf.reshape(
            chunked_value,
            (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim),
        )

        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(chunked_value),
                [batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim],
                message="Chunked value has the wrong shape",
            )

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)
        context = tf.einsum("bcwd,bcdh->bcwh", chunked_attn_probs, chunked_value)
        context = tf.transpose(
            tf.reshape(context, (batch_size, num_heads, seq_len, head_dim)),
            (0, 2, 1, 3),
        )

        return context

    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, paddings):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = tf.pad(
            hidden_states_padded, paddings
        )  # padding value is not important because it will be overwritten
        batch_size, chunk_size, seq_length, hidden_dim = shape_list(hidden_states_padded)
        hidden_states_padded = tf.reshape(hidden_states_padded, (batch_size, chunk_size, hidden_dim, seq_length))

        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        shift every row 1 step right, converting columns into diagonals.

        Example::

              chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                       -1.8348,  0.7672,  0.2986,  0.0285,
                                       -0.7584,  0.4206, -0.0405,  0.1599,
                                       2.0514, -1.1600,  0.5372,  0.2629 ]
              window_overlap = num_rows = 4
             (pad & diagonalize) =>
             [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
               0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
               0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
               0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = shape_list(chunked_hidden_states)
        paddings = tf.convert_to_tensor([[0, 0], [0, 0], [0, 0], [0, window_overlap + 1]])
        chunked_hidden_states = tf.pad(
            chunked_hidden_states, paddings
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        chunked_hidden_states = tf.reshape(
            chunked_hidden_states, (total_num_heads, num_chunks, -1)
        )  # total_num_heads x num_chunks x window_overlapL+window_overlapwindow_overlap+window_overlap
        chunked_hidden_states = chunked_hidden_states[
                                :, :, :-window_overlap
                                ]  # total_num_heads x num_chunks x window_overlapL+window_overlapwindow_overlap
        chunked_hidden_states = tf.reshape(
            chunked_hidden_states,
            (total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim),
        )  # total_num_heads x num_chunks, window_overlap x hidden_dim+window_overlap
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]

        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        batch_size, seq_length, hidden_dim = shape_list(hidden_states)
        num_output_chunks = 2 * (seq_length // (2 * window_overlap)) - 1

        # define frame size and frame stride (similar to convolution)
        frame_hop_size = window_overlap * hidden_dim
        frame_size = 2 * frame_hop_size
        hidden_states = tf.reshape(hidden_states, (batch_size, seq_length * hidden_dim))

        # chunk with overlap
        chunked_hidden_states = tf.signal.frame(hidden_states, frame_size, frame_hop_size)

        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(chunked_hidden_states),
                [batch_size, num_output_chunks, frame_size],
                message=f"Make sure chunking is correctly applied. `Chunked hidden states should have output  dimension {[batch_size, frame_size, num_output_chunks]}, but got {shape_list(chunked_hidden_states)}.",
            )

        chunked_hidden_states = tf.reshape(
            chunked_hidden_states,
            (batch_size, num_output_chunks, 2 * window_overlap, hidden_dim),
        )

        return chunked_hidden_states

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """compute global attn indices required throughout forward pass"""
        # helper variable
        num_global_attn_indices = tf.math.count_nonzero(is_index_global_attn, axis=1)
        num_global_attn_indices = tf.cast(num_global_attn_indices, dtype=tf.constant(1).dtype)

        # max number of global attn indices in batch
        max_num_global_attn_indices = tf.reduce_max(num_global_attn_indices)

        # indices of global attn
        is_index_global_attn_nonzero = tf.where(is_index_global_attn)

        # helper variable
        is_local_index_global_attn = tf.range(max_num_global_attn_indices) < tf.expand_dims(
            num_global_attn_indices, axis=-1
        )

        # location of the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = tf.where(is_local_index_global_attn)

        # location of the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = tf.where(tf.math.logical_not(is_local_index_global_attn))

        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _concat_with_global_key_attn_probs(
            self,
            attn_scores,
            key_vectors,
            query_vectors,
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
    ):
        batch_size = shape_list(key_vectors)[0]

        # select global key vectors
        global_key_vectors = tf.gather_nd(key_vectors, is_index_global_attn_nonzero)

        # create only global key vectors
        key_vectors_only_global = tf.scatter_nd(
            is_local_index_global_attn_nonzero,
            global_key_vectors,
            shape=(
                batch_size,
                max_num_global_attn_indices,
                self.num_heads,
                self.head_dim,
            ),
        )

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = tf.einsum("blhd,bshd->blhs", query_vectors, key_vectors_only_global)

        # (batch_size, max_num_global_attn_indices, seq_len, num_heads)
        attn_probs_from_global_key_trans = tf.transpose(attn_probs_from_global_key, (0, 3, 1, 2))
        mask_shape = (shape_list(is_local_index_no_global_attn_nonzero)[0],) + tuple(
            shape_list(attn_probs_from_global_key_trans)[-2:]
        )
        mask = tf.ones(mask_shape) * -10000.0
        mask = tf.cast(mask, dtype=attn_probs_from_global_key_trans.dtype)

        # scatter mask
        attn_probs_from_global_key_trans = tf.tensor_scatter_nd_update(
            attn_probs_from_global_key_trans,
            is_local_index_no_global_attn_nonzero,
            mask,
        )

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = tf.transpose(attn_probs_from_global_key_trans, (0, 2, 3, 1))

        # concat to attn_probs
        # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
        attn_scores = tf.concat((attn_probs_from_global_key, attn_scores), axis=-1)

        return attn_scores

    def _compute_attn_output_with_global_indices(
            self,
            value_vectors,
            attn_probs,
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
    ):
        batch_size = shape_list(attn_probs)[0]

        # cut local attn probs to global only
        attn_probs_only_global = attn_probs[:, :, :, :max_num_global_attn_indices]

        # select global value vectors
        global_value_vectors = tf.gather_nd(value_vectors, is_index_global_attn_nonzero)

        # create only global value vectors
        value_vectors_only_global = tf.scatter_nd(
            is_local_index_global_attn_nonzero,
            global_value_vectors,
            shape=(
                batch_size,
                max_num_global_attn_indices,
                self.num_heads,
                self.head_dim,
            ),
        )

        # compute attn output only global
        attn_output_only_global = tf.einsum("blhs,bshd->blhd", attn_probs_only_global, value_vectors_only_global)

        # reshape attn probs
        attn_probs_without_global = attn_probs[:, :, :, max_num_global_attn_indices:]

        # compute attn output with global
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors, self.one_sided_attn_window_size
        )

        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(
            self,
            attn_output,
            hidden_states,
            max_num_global_attn_indices,
            layer_head_mask,
            is_local_index_global_attn_nonzero,
            is_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
            is_index_masked,
            value_act,
            initializer_range,
            training,
    ):
        def reshape_to_matrix(input_tensor):
            """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
            ndims = input_tensor.shape.ndims
            if ndims < 2:
                raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                                 (input_tensor.shape))
            if ndims == 2:
                return input_tensor

            # width = input_tensor.shape[-1]
            width = get_shape_list(input_tensor)[-1]
            output_tensor = tf.reshape(input_tensor, [-1, width])
            return output_tensor

        batch_size, seq_len = shape_list(hidden_states)[:2]

        # prepare global hidden states
        print('-----------------------------------------')
        print(hidden_states)
        print(is_index_global_attn_nonzero)
        global_attn_hidden_states = tf.gather_nd(hidden_states, is_index_global_attn_nonzero)
        global_attn_hidden_states = tf.scatter_nd(
            is_local_index_global_attn_nonzero,
            global_attn_hidden_states,
            shape=(batch_size, max_num_global_attn_indices, self.embed_dim),
        )

        global_attn_hidden_states_2d=reshape_to_matrix(global_attn_hidden_states)
        # hidden_states_2d=reshape_to_matrix(hidden_states)
        hidden_states_2d=tf.reshape(hidden_states,[-1,self.embed_dim])


        # `value_layer` = [B*T, N*H]
        global_query_vectors_only_global = tf.layers.dense(
            global_attn_hidden_states_2d,
            self.num_heads * self.head_dim,
            activation=value_act,
            name="query_g",
            kernel_initializer=create_initializer(initializer_range))

        global_key_vectors=tf.layers.dense(
            hidden_states_2d,
            self.num_heads * self.head_dim,
            activation=value_act,
            name="key_g",
            kernel_initializer=create_initializer(initializer_range))

        global_value_vectors=tf.layers.dense(
            hidden_states_2d,
            self.num_heads * self.head_dim,
            activation=value_act,
            name="value_g",
            kernel_initializer=create_initializer(initializer_range))


        # global key, query, value
        # batch_size, -1, self.num_heads, self.head_dim
        # global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        # global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        # global_key_vectors = self.key_global(hidden_states)
        # global_value_vectors = self.value_global(hidden_states)

        # normalize
        global_query_vectors_only_global /= tf.math.sqrt(
            tf.cast(self.head_dim, dtype=global_query_vectors_only_global.dtype)
        )
        global_query_vectors_only_global = self.reshape_and_transpose(global_query_vectors_only_global, batch_size)
        global_key_vectors = self.reshape_and_transpose(global_key_vectors, batch_size)
        global_value_vectors = self.reshape_and_transpose(global_value_vectors, batch_size)

        # compute attn scores
        global_attn_scores = tf.matmul(global_query_vectors_only_global, global_key_vectors, transpose_b=True)

        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(global_attn_scores),
                [batch_size * self.num_heads, max_num_global_attn_indices, seq_len],
                message=f"global_attn_scores have the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)}, but is {shape_list(global_attn_scores)}.",
            )

        global_attn_scores = tf.reshape(
            global_attn_scores,
            (batch_size, self.num_heads, max_num_global_attn_indices, seq_len),
        )
        global_attn_scores_trans = tf.transpose(global_attn_scores, (0, 2, 1, 3))
        mask_shape = (shape_list(is_local_index_no_global_attn_nonzero)[0],) + tuple(
            shape_list(global_attn_scores_trans)[-2:]
        )
        global_attn_mask = tf.ones(mask_shape) * -10000.0
        global_attn_mask = tf.cast(global_attn_mask, dtype=global_attn_scores_trans.dtype)

        # scatter mask
        global_attn_scores_trans = tf.tensor_scatter_nd_update(
            global_attn_scores_trans,
            is_local_index_no_global_attn_nonzero,
            global_attn_mask,
        )
        global_attn_scores = tf.transpose(global_attn_scores_trans, (0, 2, 1, 3))

        # mask global attn scores
        # attn_mask = tf.tile(is_index_masked[:, None, None, :], (1, shape_list(global_attn_scores)[1], 1, 1))
        attn_mask = tf.tile(is_index_masked[:, None, None, :], (1, shape_list(global_attn_scores)[1], shape_list(global_attn_scores)[2], 1))
        temp10000=tf.zeros_like(global_attn_scores)-10000.0
        # global_attn_scores = tf.where(attn_mask, -10000.0, global_attn_scores)
        global_attn_scores = tf.where(attn_mask, temp10000, global_attn_scores)
        global_attn_scores = tf.reshape(
            global_attn_scores,
            (batch_size * self.num_heads, max_num_global_attn_indices, seq_len),
        )

        # compute global attn probs
        global_attn_probs_float = tf.nn.softmax(global_attn_scores, axis=-1)

        # apply layer head masking
        if layer_head_mask is not None:
            if tf.executing_eagerly():
                tf.debugging.assert_equal(
                    shape_list(layer_head_mask),
                    [self.num_heads],
                    message=f"Head mask for a single layer should be of size {(self.num_heads)}, but is {shape_list(layer_head_mask)}",
                )
            global_attn_probs_float = tf.reshape(layer_head_mask, (1, -1, 1, 1)) * tf.reshape(
                global_attn_probs_float, (batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
            )
            global_attn_probs_float = tf.reshape(
                global_attn_probs_float, (batch_size * self.num_heads, max_num_global_attn_indices, seq_len)
            )

        # dropout
        # global_attn_probs = self.global_dropout(global_attn_probs_float, training=training)
        global_attn_probs = tf.layers.dropout(global_attn_probs_float,rate=0.1,training=training)

        # global attn output
        global_attn_output = tf.matmul(global_attn_probs, global_value_vectors)

        if tf.executing_eagerly():
            tf.debugging.assert_equal(
                shape_list(global_attn_output),
                [batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim],
                message=f"global_attn_output tensor has the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim)}, but is {shape_list(global_attn_output)}.",
            )

        global_attn_output = tf.reshape(
            global_attn_output,
            (batch_size, self.num_heads, max_num_global_attn_indices, self.head_dim),
        )

        # get only non zero global attn output
        nonzero_global_attn_output = tf.gather_nd(
            tf.transpose(global_attn_output, (0, 2, 1, 3)),
            is_local_index_global_attn_nonzero,
        )
        nonzero_global_attn_output = tf.reshape(
            nonzero_global_attn_output,
            (shape_list(is_local_index_global_attn_nonzero)[0], -1),
        )

        # overwrite values with global attention
        attn_output = tf.tensor_scatter_nd_update(
            attn_output, is_index_global_attn_nonzero, nonzero_global_attn_output
        )

        global_attn_probs = tf.reshape(
            global_attn_probs, (batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
        )

        return attn_output, global_attn_probs

    def reshape_and_transpose(self, vector, batch_size):
        return tf.reshape(
            tf.transpose(
                tf.reshape(vector, (batch_size, -1, self.num_heads, self.head_dim)),
                (0, 2, 1, 3),
            ),
            (batch_size * self.num_heads, -1, self.head_dim),
        )

    def longformer_attention_layer(self,
                        from_tensor,
                        to_tensor,
                        attention_mask=None,
                        head_mask=None,
                        padding_len=0,
                        is_index_masked=None,
                        is_index_global_attn=None,
                        is_global_attn=False,
                        num_attention_heads=1,
                        size_per_head=512,
                        query_act=None,
                        key_act=None,
                        value_act=None,
                        attention_probs_dropout_prob=0.0,
                        initializer_range=0.02,
                        do_return_2d_tensor=False,
                        batch_size=None,
                        from_seq_length=None,
                        to_seq_length=None,
                        is_training=False):
      """Performs multi-headed attention from `from_tensor` to `to_tensor`.

      This is an implementation of multi-headed attention based on "Attention
      is all you Need". If `from_tensor` and `to_tensor` are the same, then
      this is self-attention. Each timestep in `from_tensor` attends to the
      corresponding sequence in `to_tensor`, and returns a fixed-with vector.

      This function first projects `from_tensor` into a "query" tensor and
      `to_tensor` into "key" and "value" tensors. These are (effectively) a list
      of tensors of length `num_attention_heads`, where each tensor is of shape
      [batch_size, seq_length, size_per_head].

      Then, the query and key tensors are dot-producted and scaled. These are
      softmaxed to obtain attention probabilities. The value tensors are then
      interpolated by these probabilities, then concatenated back to a single
      tensor and returned.

      In practice, the multi-headed attention are done with transposes and
      reshapes rather than actual separate tensors.

      Args:
        from_tensor: float Tensor of shape [batch_size, from_seq_length,
          from_width].
        to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
        attention_mask: (optional) int32 Tensor of shape [batch_size,
          from_seq_length, to_seq_length]. The values should be 1 or 0. The
          attention scores will effectively be set to -infinity for any positions in
          the mask that are 0, and will be unchanged for positions that are 1.
        num_attention_heads: int. Number of attention heads.
        size_per_head: int. Size of each attention head.
        query_act: (optional) Activation function for the query transform.
        key_act: (optional) Activation function for the key transform.
        value_act: (optional) Activation function for the value transform.
        attention_probs_dropout_prob: (optional) float. Dropout probability of the
          attention probabilities.
        initializer_range: float. Range of the weight initializer.
        do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
          * from_seq_length, num_attention_heads * size_per_head]. If False, the
          output will be of shape [batch_size, from_seq_length, num_attention_heads
          * size_per_head].
        batch_size: (Optional) int. If the input is 2D, this might be the batch size
          of the 3D version of the `from_tensor` and `to_tensor`.
        from_seq_length: (Optional) If the input is 2D, this might be the seq length
          of the 3D version of the `from_tensor`.
        to_seq_length: (Optional) If the input is 2D, this might be the seq length
          of the 3D version of the `to_tensor`.

      Returns:
        float Tensor of shape [batch_size, from_seq_length,
          num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
          true, this will be of shape [batch_size * from_seq_length,
          num_attention_heads * size_per_head]).

      Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
      """
      self.num_heads = num_attention_heads
      self.head_dim = size_per_head
      self.embed_dim=num_attention_heads*size_per_head

      def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                               seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

      from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
      to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

      if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

      if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
      elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
          raise ValueError(
              "When passing in rank 2 tensors to attention_layer, the values "
              "for `batch_size`, `from_seq_length`, and `to_seq_length` "
              "must all be specified.")

      # Scalar dimensions referenced here:
      #   B = batch size (number of sequences)
      #   F = `from_tensor` sequence length
      #   T = `to_tensor` sequence length
      #   N = `num_attention_heads`
      #   H = `size_per_head`

      from_tensor_2d = reshape_to_matrix(from_tensor)
      to_tensor_2d = reshape_to_matrix(to_tensor)

      # `query_layer` = [B*F, N*H]
      query_layer = tf.layers.dense(
          from_tensor_2d,
          num_attention_heads * size_per_head,
          activation=query_act,
          name="query",
          kernel_initializer=create_initializer(initializer_range))

      # `key_layer` = [B*T, N*H]
      key_layer = tf.layers.dense(
          to_tensor_2d,
          num_attention_heads * size_per_head,
          activation=key_act,
          name="key",
          kernel_initializer=create_initializer(initializer_range))

      # `value_layer` = [B*T, N*H]
      value_layer = tf.layers.dense(
          to_tensor_2d,
          num_attention_heads * size_per_head,
          activation=value_act,
          name="value",
          kernel_initializer=create_initializer(initializer_range))

      # `query_layer` = [B, N, F, H]
      query_layer = transpose_for_scores(query_layer, batch_size,
                                         num_attention_heads, from_seq_length,
                                         size_per_head)

      # `key_layer` = [B, N, T, H]
      key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                       to_seq_length, size_per_head)

      # normalize query
      query_layer = tf.multiply(query_layer,
                                     1.0 / math.sqrt(float(size_per_head)))
      # `query_vectors` = [B, F, N, H]
      query_vectors=tf.transpose(query_layer, [0, 2, 1, 3])
      # `key_vectors` = [B, T, N, H]
      key_vectors=tf.transpose(key_layer, [0, 2, 1, 3])

      # attn_scores = (batch_size, seq_len, num_heads, window*2+1)
      attn_scores = self._sliding_chunks_query_key_matmul(
          query_vectors, key_vectors, self.one_sided_attn_window_size
      )

      # diagonal mask with zeros everywhere and -inf inplace of padding
      diagonal_mask = self._sliding_chunks_query_key_matmul(
          tf.ones(shape_list(attention_mask)),
          attention_mask,
          self.one_sided_attn_window_size,
      )

      # pad local attention probs
      attn_scores += diagonal_mask

      if tf.executing_eagerly():
          tf.debugging.assert_equal(
              shape_list(attn_scores),
              [batch_size, to_seq_length, self.num_heads, self.one_sided_attn_window_size * 2 + 1],
              message=f"attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {shape_list(attn_scores)}",
          )

      # compute global attn indices required through out forward fn
      (
          max_num_global_attn_indices,
          is_index_global_attn_nonzero,
          is_local_index_global_attn_nonzero,
          is_local_index_no_global_attn_nonzero,
      ) = self._get_global_attn_indices(is_index_global_attn)
      max_num_global_attn_indices=1
      # is_global_attn=tf.constant(True,tf.bool)

      # this function is only relevant for global attention
      # attn_scores = tf.cond(
      #     is_global_attn,
      #     lambda: self._concat_with_global_key_attn_probs(
      #         attn_scores=attn_scores,
      #         query_vectors=query_vectors,
      #         key_vectors=key_vectors,
      #         max_num_global_attn_indices=max_num_global_attn_indices,
      #         is_index_global_attn_nonzero=is_index_global_attn_nonzero,
      #         is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
      #         is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
      #     ),
      #     lambda: attn_scores,
      # )

      attn_scores = self._concat_with_global_key_attn_probs(
              attn_scores=attn_scores,
              query_vectors=query_vectors,
              key_vectors=key_vectors,
              max_num_global_attn_indices=max_num_global_attn_indices,
              is_index_global_attn_nonzero=is_index_global_attn_nonzero,
              is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
              is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
          )
      attn_probs = tf.nn.softmax(attn_scores, axis=-1)

      # softmax sometimes inserts NaN if all positions are masked, replace them with 0
      # Make sure to create a mask with the proper shape:
      # if is_global_attn==True => [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1]
      # if is_global_attn==False => [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + 1]
      # masked_index = tf.cond(
      #     is_global_attn,
      #     lambda: tf.tile(
      #         is_index_masked[:, :, None, None],
      #         (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1),
      #     ),
      #     lambda: tf.tile(
      #         is_index_masked[:, :, None, None],
      #         (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + 1),
      #     ),
      # )
      masked_index = tf.tile(
              is_index_masked[:, :, None, None],
              (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1),
          )
      # print('**************************************')
      # print(masked_index)
      # print(attn_probs)

      # if tf.executing_eagerly():
      #     tf.debugging.assert_equal(
      #         shape_list(masked_index),
      #         [batch_size, to_seq_length, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1],
      #         message=f"masked_index should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1}), but is of size {shape_list(masked_index)}",
      #     )
      # if tf.executing_eagerly():
      #     tf.debugging.assert_equal(
      #         shape_list(attn_probs),
      #         [batch_size, to_seq_length, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1],
      #         message=f"attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1}), but is of size {shape_list(attn_probs)}",
      #     )
      masked_shape=[batch_size, to_seq_length, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1]
      # attn_probs=attn_probs+tf.zeros(masked_shape, dtype=attn_probs.dtype)
      # attn_probs=attn_probs+tf.zeros(shape_list(masked_index), dtype=attn_probs.dtype)
      # attn_probs=tf.zeros(masked_shape, dtype=attn_probs.dtype)+tf.zeros(shape_list(masked_index), dtype=attn_probs.dtype)
      # attn_probs=tf.zeros(masked_shape, dtype=masked_index.dtype)+masked_index

      attn_probs = tf.where(
          masked_index,
          tf.zeros(shape_list(masked_index), dtype=attn_probs.dtype),
          # tf.zeros(masked_shape, dtype=attn_probs.dtype),
          # tf.zeros_like(attn_probs),
          # tf.zeros_like(masked_index),
          attn_probs,
      )
      # attn_probs=tf.reshape(attn_probs,masked_shape)

      # apply dropout
      # attn_probs = dropout_TF(attn_probs, attention_probs_dropout_prob,is_training=is_training)
      attn_probs = tf.layers.dropout(attn_probs, attention_probs_dropout_prob,training=is_training)
      # attn_probs = self.dropout(attn_probs, training=is_training)
      value_vectors = tf.reshape(value_layer, (batch_size, to_seq_length, self.num_heads, self.head_dim))

      # if global attention, compute sum of global and local attn
      # attn_output = tf.cond(
      #     is_global_attn,
      #     lambda: self._compute_attn_output_with_global_indices(
      #         value_vectors=value_vectors,
      #         attn_probs=attn_probs,
      #         max_num_global_attn_indices=max_num_global_attn_indices,
      #         is_index_global_attn_nonzero=is_index_global_attn_nonzero,
      #         is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
      #     ),
      #     lambda: self._sliding_chunks_matmul_attn_probs_value(
      #         attn_probs, value_vectors, self.one_sided_attn_window_size
      #     ),
      # )
      #
      #tf.cond 在服务器版本中难以进行不同shape的输出，会与之后的静态图产生冲突
      attn_output = self._compute_attn_output_with_global_indices(
              value_vectors=value_vectors,
              attn_probs=attn_probs,
              max_num_global_attn_indices=max_num_global_attn_indices,
              is_index_global_attn_nonzero=is_index_global_attn_nonzero,
              is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
          )

      if tf.executing_eagerly():
          tf.debugging.assert_equal(
              shape_list(attn_output),
              [batch_size, to_seq_length, self.num_heads, self.head_dim],
              message="Unexpected size",
          )
      attn_output = tf.reshape(attn_output, (batch_size, from_seq_length, self.embed_dim))

      hidden_states=tf.reshape(from_tensor,[batch_size,from_seq_length,-1])
      # compute value for global attention and overwrite to attention output
      # TODO: remove the redundant computation
      # attn_output, global_attn_probs = tf.cond(
      #     is_global_attn,
      #     lambda: self._compute_global_attn_output_from_hidden(
      #         attn_output=attn_output,
      #         hidden_states=hidden_states,
      #         max_num_global_attn_indices=max_num_global_attn_indices,
      #         layer_head_mask=head_mask,
      #         is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
      #         is_index_global_attn_nonzero=is_index_global_attn_nonzero,
      #         is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
      #         is_index_masked=is_index_masked,
      #         value_act=value_act,
      #         initializer_range=initializer_range,
      #         training=is_training,
      #     ),
      #     lambda: (attn_output, tf.zeros((batch_size, self.num_heads, max_num_global_attn_indices, from_seq_length))),
      # )

      if do_return_2d_tensor:
          attn_output = tf.reshape(attn_output, (batch_size*from_seq_length, self.embed_dim))
      else:
          attn_output = tf.reshape(attn_output, (batch_size,from_seq_length, self.embed_dim))

      context_layer=attn_output

      # # Take the dot product between "query" and "key" to get the raw
      # # attention scores.
      # #batch_size, num_attention_heads, from_seq_length, to_seq_length
      # # `attention_scores` = [B, N, F, T]
      # attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
      # attention_scores = tf.multiply(attention_scores,
      #                                1.0 / math.sqrt(float(size_per_head)))
      #
      # if attention_mask is not None:
      #   # `attention_mask` = [B, 1, F, T]
      #   attention_mask = tf.expand_dims(attention_mask, axis=[1])
      #
      #   # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      #   # masked positions, this operation will create a tensor which is 0.0 for
      #   # positions we want to attend and -10000.0 for masked positions.
      #   adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
      #
      #   # Since we are adding it to the raw scores before the softmax, this is
      #   # effectively the same as removing these entirely.
      #   attention_scores += adder
      #
      # # Normalize the attention scores to probabilities.
      # # `attention_probs` = [B, N, F, T]
      # attention_probs = tf.nn.softmax(attention_scores)
      #
      # # This is actually dropping out entire tokens to attend to, which might
      # # seem a bit unusual, but is taken from the original Transformer paper.
      # attention_probs = dropout(attention_probs, attention_probs_dropout_prob)
      #
      # # `value_layer` = [B, T, N, H]
      # value_layer = tf.reshape(
      #     value_layer,
      #     [batch_size, to_seq_length, num_attention_heads, size_per_head])
      #
      # # `value_layer` = [B, N, T, H]
      # value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
      #
      # # `context_layer` = [B, N, F, H]
      # context_layer = tf.matmul(attention_probs, value_layer)
      #
      # # `context_layer` = [B, F, N, H]
      # context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
      #
      # if do_return_2d_tensor:
      #   # `context_layer` = [B*F, N*H]
      #   context_layer = tf.reshape(
      #       context_layer,
      #       [batch_size * from_seq_length, num_attention_heads * size_per_head])
      # else:
      #   # `context_layer` = [B, F, N*H]
      #   context_layer = tf.reshape(
      #       context_layer,
      #       [batch_size, from_seq_length, num_attention_heads * size_per_head])

      return context_layer

    def longformer_global_overwrite_attention_layer(self,
                        from_tensor,
                        to_tensor,
                        attention_mask=None,
                        head_mask=None,
                        padding_len=0,
                        is_index_masked=None,
                        is_index_global_attn=None,
                        is_global_attn=False,
                        num_attention_heads=1,
                        size_per_head=512,
                        query_act=None,
                        key_act=None,
                        value_act=None,
                        attention_probs_dropout_prob=0.0,
                        initializer_range=0.02,
                        do_return_2d_tensor=False,
                        batch_size=None,
                        from_seq_length=None,
                        to_seq_length=None,
                        is_training=False):
      """Performs multi-headed attention from `from_tensor` to `to_tensor`.

      This is an implementation of multi-headed attention based on "Attention
      is all you Need". If `from_tensor` and `to_tensor` are the same, then
      this is self-attention. Each timestep in `from_tensor` attends to the
      corresponding sequence in `to_tensor`, and returns a fixed-with vector.

      This function first projects `from_tensor` into a "query" tensor and
      `to_tensor` into "key" and "value" tensors. These are (effectively) a list
      of tensors of length `num_attention_heads`, where each tensor is of shape
      [batch_size, seq_length, size_per_head].

      Then, the query and key tensors are dot-producted and scaled. These are
      softmaxed to obtain attention probabilities. The value tensors are then
      interpolated by these probabilities, then concatenated back to a single
      tensor and returned.

      In practice, the multi-headed attention are done with transposes and
      reshapes rather than actual separate tensors.

      Args:
        from_tensor: float Tensor of shape [batch_size, from_seq_length,
          from_width].
        to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
        attention_mask: (optional) int32 Tensor of shape [batch_size,
          from_seq_length, to_seq_length]. The values should be 1 or 0. The
          attention scores will effectively be set to -infinity for any positions in
          the mask that are 0, and will be unchanged for positions that are 1.
        num_attention_heads: int. Number of attention heads.
        size_per_head: int. Size of each attention head.
        query_act: (optional) Activation function for the query transform.
        key_act: (optional) Activation function for the key transform.
        value_act: (optional) Activation function for the value transform.
        attention_probs_dropout_prob: (optional) float. Dropout probability of the
          attention probabilities.
        initializer_range: float. Range of the weight initializer.
        do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
          * from_seq_length, num_attention_heads * size_per_head]. If False, the
          output will be of shape [batch_size, from_seq_length, num_attention_heads
          * size_per_head].
        batch_size: (Optional) int. If the input is 2D, this might be the batch size
          of the 3D version of the `from_tensor` and `to_tensor`.
        from_seq_length: (Optional) If the input is 2D, this might be the seq length
          of the 3D version of the `from_tensor`.
        to_seq_length: (Optional) If the input is 2D, this might be the seq length
          of the 3D version of the `to_tensor`.

      Returns:
        float Tensor of shape [batch_size, from_seq_length,
          num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
          true, this will be of shape [batch_size * from_seq_length,
          num_attention_heads * size_per_head]).

      Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
      """
      self.num_heads = num_attention_heads
      self.head_dim = size_per_head
      self.embed_dim=num_attention_heads*size_per_head

      def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                               seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

      from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
      to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

      if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

      if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
      elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
          raise ValueError(
              "When passing in rank 2 tensors to attention_layer, the values "
              "for `batch_size`, `from_seq_length`, and `to_seq_length` "
              "must all be specified.")

      # Scalar dimensions referenced here:
      #   B = batch size (number of sequences)
      #   F = `from_tensor` sequence length
      #   T = `to_tensor` sequence length
      #   N = `num_attention_heads`
      #   H = `size_per_head`

      from_tensor_2d = reshape_to_matrix(from_tensor)
      to_tensor_2d = reshape_to_matrix(to_tensor)

      # `query_layer` = [B*F, N*H]
      query_layer = tf.layers.dense(
          from_tensor_2d,
          num_attention_heads * size_per_head,
          activation=query_act,
          name="query",
          kernel_initializer=create_initializer(initializer_range))

      # `key_layer` = [B*T, N*H]
      key_layer = tf.layers.dense(
          to_tensor_2d,
          num_attention_heads * size_per_head,
          activation=key_act,
          name="key",
          kernel_initializer=create_initializer(initializer_range))

      # `value_layer` = [B*T, N*H]
      value_layer = tf.layers.dense(
          to_tensor_2d,
          num_attention_heads * size_per_head,
          activation=value_act,
          name="value",
          kernel_initializer=create_initializer(initializer_range))

      # `query_layer` = [B, N, F, H]
      query_layer = transpose_for_scores(query_layer, batch_size,
                                         num_attention_heads, from_seq_length,
                                         size_per_head)

      # `key_layer` = [B, N, T, H]
      key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                       to_seq_length, size_per_head)

      # normalize query
      query_layer = tf.multiply(query_layer,
                                     1.0 / math.sqrt(float(size_per_head)))
      # `query_vectors` = [B, F, N, H]
      query_vectors=tf.transpose(query_layer, [0, 2, 1, 3])
      # `key_vectors` = [B, T, N, H]
      key_vectors=tf.transpose(key_layer, [0, 2, 1, 3])

      # attn_scores = (batch_size, seq_len, num_heads, window*2+1)
      attn_scores = self._sliding_chunks_query_key_matmul(
          query_vectors, key_vectors, self.one_sided_attn_window_size
      )

      # diagonal mask with zeros everywhere and -inf inplace of padding
      diagonal_mask = self._sliding_chunks_query_key_matmul(
          tf.ones(shape_list(attention_mask)),
          attention_mask,
          self.one_sided_attn_window_size,
      )

      # pad local attention probs
      attn_scores += diagonal_mask

      if tf.executing_eagerly():
          tf.debugging.assert_equal(
              shape_list(attn_scores),
              [batch_size, to_seq_length, self.num_heads, self.one_sided_attn_window_size * 2 + 1],
              message=f"attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {shape_list(attn_scores)}",
          )

      # compute global attn indices required through out forward fn
      (
          max_num_global_attn_indices,
          is_index_global_attn_nonzero,
          is_local_index_global_attn_nonzero,
          is_local_index_no_global_attn_nonzero,
      ) = self._get_global_attn_indices(is_index_global_attn)
      # max_num_global_attn_indices=1
      # is_global_attn=tf.constant(True,tf.bool)

      # this function is only relevant for global attention
      # attn_scores = tf.cond(
      #     is_global_attn,
      #     lambda: self._concat_with_global_key_attn_probs(
      #         attn_scores=attn_scores,
      #         query_vectors=query_vectors,
      #         key_vectors=key_vectors,
      #         max_num_global_attn_indices=max_num_global_attn_indices,
      #         is_index_global_attn_nonzero=is_index_global_attn_nonzero,
      #         is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
      #         is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
      #     ),
      #     lambda: attn_scores,
      # )

      attn_scores = self._concat_with_global_key_attn_probs(
              attn_scores=attn_scores,
              query_vectors=query_vectors,
              key_vectors=key_vectors,
              max_num_global_attn_indices=max_num_global_attn_indices,
              is_index_global_attn_nonzero=is_index_global_attn_nonzero,
              is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
              is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
          )
      attn_probs = tf.nn.softmax(attn_scores, axis=-1)

      # softmax sometimes inserts NaN if all positions are masked, replace them with 0
      # Make sure to create a mask with the proper shape:
      # if is_global_attn==True => [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1]
      # if is_global_attn==False => [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + 1]
      # masked_index = tf.cond(
      #     is_global_attn,
      #     lambda: tf.tile(
      #         is_index_masked[:, :, None, None],
      #         (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1),
      #     ),
      #     lambda: tf.tile(
      #         is_index_masked[:, :, None, None],
      #         (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + 1),
      #     ),
      # )
      masked_index = tf.tile(
              is_index_masked[:, :, None, None],
              (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1),
          )
      # print('**************************************')
      # print(masked_index)
      # print(attn_probs)

      # if tf.executing_eagerly():
      #     tf.debugging.assert_equal(
      #         shape_list(masked_index),
      #         [batch_size, to_seq_length, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1],
      #         message=f"masked_index should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1}), but is of size {shape_list(masked_index)}",
      #     )
      # if tf.executing_eagerly():
      #     tf.debugging.assert_equal(
      #         shape_list(attn_probs),
      #         [batch_size, to_seq_length, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1],
      #         message=f"attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1}), but is of size {shape_list(attn_probs)}",
      #     )
      masked_shape=[batch_size, to_seq_length, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1]
      # attn_probs=attn_probs+tf.zeros(masked_shape, dtype=attn_probs.dtype)
      # attn_probs=attn_probs+tf.zeros(shape_list(masked_index), dtype=attn_probs.dtype)
      # attn_probs=tf.zeros(masked_shape, dtype=attn_probs.dtype)+tf.zeros(shape_list(masked_index), dtype=attn_probs.dtype)
      # attn_probs=tf.zeros(masked_shape, dtype=masked_index.dtype)+masked_index

      attn_probs = tf.where(
          masked_index,
          tf.zeros(shape_list(masked_index), dtype=attn_probs.dtype),
          # tf.zeros(masked_shape, dtype=attn_probs.dtype),
          # tf.zeros_like(attn_probs),
          # tf.zeros_like(masked_index),
          attn_probs,
      )
      # attn_probs=tf.reshape(attn_probs,masked_shape)

      # apply dropout
      # attn_probs = dropout_TF(attn_probs, attention_probs_dropout_prob,is_training=is_training)
      attn_probs = tf.layers.dropout(attn_probs, attention_probs_dropout_prob,training=is_training)
      # attn_probs = self.dropout(attn_probs, training=is_training)
      value_vectors = tf.reshape(value_layer, (batch_size, to_seq_length, self.num_heads, self.head_dim))

      # if global attention, compute sum of global and local attn
      # attn_output = tf.cond(
      #     is_global_attn,
      #     lambda: self._compute_attn_output_with_global_indices(
      #         value_vectors=value_vectors,
      #         attn_probs=attn_probs,
      #         max_num_global_attn_indices=max_num_global_attn_indices,
      #         is_index_global_attn_nonzero=is_index_global_attn_nonzero,
      #         is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
      #     ),
      #     lambda: self._sliding_chunks_matmul_attn_probs_value(
      #         attn_probs, value_vectors, self.one_sided_attn_window_size
      #     ),
      # )
      #
      #tf.cond 在服务器版本中难以进行不同shape的输出，会与之后的静态图产生冲突
      attn_output = self._compute_attn_output_with_global_indices(
              value_vectors=value_vectors,
              attn_probs=attn_probs,
              max_num_global_attn_indices=max_num_global_attn_indices,
              is_index_global_attn_nonzero=is_index_global_attn_nonzero,
              is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
          )

      if tf.executing_eagerly():
          tf.debugging.assert_equal(
              shape_list(attn_output),
              [batch_size, to_seq_length, self.num_heads, self.head_dim],
              message="Unexpected size",
          )
      attn_output = tf.reshape(attn_output, (batch_size, from_seq_length, self.embed_dim))

      hidden_states=tf.reshape(from_tensor,[batch_size,from_seq_length,-1])
      # compute value for global attention and overwrite to attention output
      # TODO: remove the redundant computation
      # attn_output, global_attn_probs = tf.cond(
      #     is_global_attn,
      #     lambda: self._compute_global_attn_output_from_hidden(
      #         attn_output=attn_output,
      #         hidden_states=hidden_states,
      #         max_num_global_attn_indices=max_num_global_attn_indices,
      #         layer_head_mask=head_mask,
      #         is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
      #         is_index_global_attn_nonzero=is_index_global_attn_nonzero,
      #         is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
      #         is_index_masked=is_index_masked,
      #         value_act=value_act,
      #         initializer_range=initializer_range,
      #         training=is_training,
      #     ),
      #     lambda: (attn_output, tf.zeros((batch_size, self.num_heads, max_num_global_attn_indices, from_seq_length))),
      # )
      attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(
              attn_output=attn_output,
              hidden_states=hidden_states,
              max_num_global_attn_indices=max_num_global_attn_indices,
              layer_head_mask=head_mask,
              is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
              is_index_global_attn_nonzero=is_index_global_attn_nonzero,
              is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
              is_index_masked=is_index_masked,
              value_act=value_act,
              initializer_range=initializer_range,
              training=is_training,
          )

      if do_return_2d_tensor:
          attn_output = tf.reshape(attn_output, (batch_size*from_seq_length, self.embed_dim))
      else:
          attn_output = tf.reshape(attn_output, (batch_size,from_seq_length, self.embed_dim))

      context_layer=attn_output
      #
      # if do_return_2d_tensor:
      #   # `context_layer` = [B*F, N*H]
      #   context_layer = tf.reshape(
      #       context_layer,
      #       [batch_size * from_seq_length, num_attention_heads * size_per_head])
      # else:
      #   # `context_layer` = [B, F, N*H]
      #   context_layer = tf.reshape(
      #       context_layer,
      #       [batch_size, from_seq_length, num_attention_heads * size_per_head])

      return context_layer

    def longformer_slid_attention_layer(self,
                        from_tensor,
                        to_tensor,
                        attention_mask=None,
                        head_mask=None,
                        is_index_masked=None,
                        padding_len=0,
                        num_attention_heads=1,
                        size_per_head=512,
                        query_act=None,
                        key_act=None,
                        value_act=None,
                        attention_probs_dropout_prob=0.0,
                        initializer_range=0.02,
                        do_return_2d_tensor=False,
                        batch_size=None,
                        from_seq_length=None,
                        to_seq_length=None,
                        is_training=False):
      """Performs multi-headed attention from `from_tensor` to `to_tensor`.

      This is an implementation of multi-headed attention based on "Attention
      is all you Need". If `from_tensor` and `to_tensor` are the same, then
      this is self-attention. Each timestep in `from_tensor` attends to the
      corresponding sequence in `to_tensor`, and returns a fixed-with vector.

      This function first projects `from_tensor` into a "query" tensor and
      `to_tensor` into "key" and "value" tensors. These are (effectively) a list
      of tensors of length `num_attention_heads`, where each tensor is of shape
      [batch_size, seq_length, size_per_head].

      Then, the query and key tensors are dot-producted and scaled. These are
      softmaxed to obtain attention probabilities. The value tensors are then
      interpolated by these probabilities, then concatenated back to a single
      tensor and returned.

      In practice, the multi-headed attention are done with transposes and
      reshapes rather than actual separate tensors.

      Args:
        from_tensor: float Tensor of shape [batch_size, from_seq_length,
          from_width].
        to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
        attention_mask: (optional) int32 Tensor of shape [batch_size,
          from_seq_length, to_seq_length]. The values should be 1 or 0. The
          attention scores will effectively be set to -infinity for any positions in
          the mask that are 0, and will be unchanged for positions that are 1.
        num_attention_heads: int. Number of attention heads.
        size_per_head: int. Size of each attention head.
        query_act: (optional) Activation function for the query transform.
        key_act: (optional) Activation function for the key transform.
        value_act: (optional) Activation function for the value transform.
        attention_probs_dropout_prob: (optional) float. Dropout probability of the
          attention probabilities.
        initializer_range: float. Range of the weight initializer.
        do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
          * from_seq_length, num_attention_heads * size_per_head]. If False, the
          output will be of shape [batch_size, from_seq_length, num_attention_heads
          * size_per_head].
        batch_size: (Optional) int. If the input is 2D, this might be the batch size
          of the 3D version of the `from_tensor` and `to_tensor`.
        from_seq_length: (Optional) If the input is 2D, this might be the seq length
          of the 3D version of the `from_tensor`.
        to_seq_length: (Optional) If the input is 2D, this might be the seq length
          of the 3D version of the `to_tensor`.

      Returns:
        float Tensor of shape [batch_size, from_seq_length,
          num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
          true, this will be of shape [batch_size * from_seq_length,
          num_attention_heads * size_per_head]).

      Raises:
        ValueError: Any of the arguments or tensor shapes are invalid.
      """
      self.num_heads = num_attention_heads
      self.head_dim = size_per_head
      self.embed_dim=num_attention_heads*size_per_head

      def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                               seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

      from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
      to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

      if len(from_shape) != len(to_shape):
        raise ValueError(
            "The rank of `from_tensor` must match the rank of `to_tensor`.")

      if len(from_shape) == 3:
        batch_size = from_shape[0]
        from_seq_length = from_shape[1]
        to_seq_length = to_shape[1]
      elif len(from_shape) == 2:
        if (batch_size is None or from_seq_length is None or to_seq_length is None):
          raise ValueError(
              "When passing in rank 2 tensors to attention_layer, the values "
              "for `batch_size`, `from_seq_length`, and `to_seq_length` "
              "must all be specified.")

      # Scalar dimensions referenced here:
      #   B = batch size (number of sequences)
      #   F = `from_tensor` sequence length
      #   T = `to_tensor` sequence length
      #   N = `num_attention_heads`
      #   H = `size_per_head`

      from_tensor_2d = reshape_to_matrix(from_tensor)
      to_tensor_2d = reshape_to_matrix(to_tensor)

      # `query_layer` = [B*F, N*H]
      query_layer = tf.layers.dense(
          from_tensor_2d,
          num_attention_heads * size_per_head,
          activation=query_act,
          name="query",
          kernel_initializer=create_initializer(initializer_range))

      # `key_layer` = [B*T, N*H]
      key_layer = tf.layers.dense(
          to_tensor_2d,
          num_attention_heads * size_per_head,
          activation=key_act,
          name="key",
          kernel_initializer=create_initializer(initializer_range))

      # `value_layer` = [B*T, N*H]
      value_layer = tf.layers.dense(
          to_tensor_2d,
          num_attention_heads * size_per_head,
          activation=value_act,
          name="value",
          kernel_initializer=create_initializer(initializer_range))

      # `query_layer` = [B, N, F, H]
      query_layer = transpose_for_scores(query_layer, batch_size,
                                         num_attention_heads, from_seq_length,
                                         size_per_head)

      # `key_layer` = [B, N, T, H]
      key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                       to_seq_length, size_per_head)

      # normalize query
      query_layer = tf.multiply(query_layer,
                                     1.0 / math.sqrt(float(size_per_head)))
      # `query_vectors` = [B, F, N, H]
      query_vectors=tf.transpose(query_layer, [0, 2, 1, 3])
      # `key_vectors` = [B, T, N, H]
      key_vectors=tf.transpose(key_layer, [0, 2, 1, 3])

      # attn_scores = (batch_size, seq_len, num_heads, window*2+1)
      attn_scores = self._sliding_chunks_query_key_matmul(
          query_vectors, key_vectors, self.one_sided_attn_window_size
      )

      # diagonal mask with zeros everywhere and -inf inplace of padding
      diagonal_mask = self._sliding_chunks_query_key_matmul(
          tf.ones(shape_list(attention_mask)),
          attention_mask,
          self.one_sided_attn_window_size,
      )

      # pad local attention probs
      attn_scores += diagonal_mask

      if tf.executing_eagerly():
          tf.debugging.assert_equal(
              shape_list(attn_scores),
              [batch_size, to_seq_length, self.num_heads, self.one_sided_attn_window_size * 2 + 1],
              message=f"attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {shape_list(attn_scores)}",
          )

      # compute global attn indices required through out forward fn
      # (
      #     max_num_global_attn_indices,
      #     is_index_global_attn_nonzero,
      #     is_local_index_global_attn_nonzero,
      #     is_local_index_no_global_attn_nonzero,
      # ) = self._get_global_attn_indices(is_index_global_attn)
      # max_num_global_attn_indices=1
      # is_global_attn=tf.constant(True,tf.bool)

      # this function is only relevant for global attention
      # attn_scores = tf.cond(
      #     is_global_attn,
      #     lambda: self._concat_with_global_key_attn_probs(
      #         attn_scores=attn_scores,
      #         query_vectors=query_vectors,
      #         key_vectors=key_vectors,
      #         max_num_global_attn_indices=max_num_global_attn_indices,
      #         is_index_global_attn_nonzero=is_index_global_attn_nonzero,
      #         is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
      #         is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
      #     ),
      #     lambda: attn_scores,
      # )
      #
      # attn_scores = self._concat_with_global_key_attn_probs(
      #         attn_scores=attn_scores,
      #         query_vectors=query_vectors,
      #         key_vectors=key_vectors,
      #         max_num_global_attn_indices=max_num_global_attn_indices,
      #         is_index_global_attn_nonzero=is_index_global_attn_nonzero,
      #         is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
      #         is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
      #     )
      attn_probs = tf.nn.softmax(attn_scores, axis=-1)

      # softmax sometimes inserts NaN if all positions are masked, replace them with 0
      # Make sure to create a mask with the proper shape:
      # if is_global_attn==True => [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1]
      # if is_global_attn==False => [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + 1]
      # masked_index = tf.cond(
      #     is_global_attn,
      #     lambda: tf.tile(
      #         is_index_masked[:, :, None, None],
      #         (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1),
      #     ),
      #     lambda: tf.tile(
      #         is_index_masked[:, :, None, None],
      #         (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + 1),
      #     ),
      # )
      masked_index = tf.tile(
              is_index_masked[:, :, None, None],
              (1, 1, self.num_heads, self.one_sided_attn_window_size * 2 + 1),
          )
      # print('**************************************')
      # print(masked_index)
      # print(attn_probs)

      # if tf.executing_eagerly():
      #     tf.debugging.assert_equal(
      #         shape_list(masked_index),
      #         [batch_size, to_seq_length, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1],
      #         message=f"masked_index should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1}), but is of size {shape_list(masked_index)}",
      #     )
      # if tf.executing_eagerly():
      #     tf.debugging.assert_equal(
      #         shape_list(attn_probs),
      #         [batch_size, to_seq_length, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1],
      #         message=f"attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1}), but is of size {shape_list(attn_probs)}",
      #     )
      # masked_shape=[batch_size, to_seq_length, self.num_heads, self.one_sided_attn_window_size * 2 + max_num_global_attn_indices + 1]
      # attn_probs=attn_probs+tf.zeros(masked_shape, dtype=attn_probs.dtype)
      # attn_probs=attn_probs+tf.zeros(shape_list(masked_index), dtype=attn_probs.dtype)
      # attn_probs=tf.zeros(masked_shape, dtype=attn_probs.dtype)+tf.zeros(shape_list(masked_index), dtype=attn_probs.dtype)
      # attn_probs=tf.zeros(masked_shape, dtype=masked_index.dtype)+masked_index

      attn_probs = tf.where(
          masked_index,
          tf.zeros(shape_list(masked_index), dtype=attn_probs.dtype),
          # tf.zeros(masked_shape, dtype=attn_probs.dtype),
          # tf.zeros_like(attn_probs),
          # tf.zeros_like(masked_index),
          attn_probs,
      )
      # attn_probs=tf.reshape(attn_probs,masked_shape)

      # apply dropout
      # attn_probs = dropout_TF(attn_probs, attention_probs_dropout_prob,is_training=is_training)
      attn_probs = tf.layers.dropout(attn_probs, attention_probs_dropout_prob,training=is_training)
      # attn_probs = self.dropout(attn_probs, training=is_training)
      value_vectors = tf.reshape(value_layer, (batch_size, to_seq_length, self.num_heads, self.head_dim))

      # if global attention, compute sum of global and local attn
      # attn_output = tf.cond(
      #     is_global_attn,
      #     lambda: self._compute_attn_output_with_global_indices(
      #         value_vectors=value_vectors,
      #         attn_probs=attn_probs,
      #         max_num_global_attn_indices=max_num_global_attn_indices,
      #         is_index_global_attn_nonzero=is_index_global_attn_nonzero,
      #         is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
      #     ),
      #     lambda: self._sliding_chunks_matmul_attn_probs_value(
      #         attn_probs, value_vectors, self.one_sided_attn_window_size
      #     ),
      # )
      #
      #tf.cond 在服务器版本中难以进行不同shape的输出，会与之后的静态图产生冲突
      attn_output = self._sliding_chunks_matmul_attn_probs_value(
              attn_probs, value_vectors, self.one_sided_attn_window_size
          )

      if tf.executing_eagerly():
          tf.debugging.assert_equal(
              shape_list(attn_output),
              [batch_size, to_seq_length, self.num_heads, self.head_dim],
              message="Unexpected size",
          )
      attn_output = tf.reshape(attn_output, (batch_size, from_seq_length, self.embed_dim))

      hidden_states=tf.reshape(from_tensor,[batch_size,from_seq_length,-1])
      # compute value for global attention and overwrite to attention output
      # TODO: remove the redundant computation
      # attn_output, global_attn_probs = tf.cond(
      #     is_global_attn,
      #     lambda: self._compute_global_attn_output_from_hidden(
      #         attn_output=attn_output,
      #         hidden_states=hidden_states,
      #         max_num_global_attn_indices=max_num_global_attn_indices,
      #         layer_head_mask=head_mask,
      #         is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
      #         is_index_global_attn_nonzero=is_index_global_attn_nonzero,
      #         is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
      #         is_index_masked=is_index_masked,
      #         value_act=value_act,
      #         initializer_range=initializer_range,
      #         training=is_training,
      #     ),
      #     lambda: (attn_output, tf.zeros((batch_size, self.num_heads, max_num_global_attn_indices, from_seq_length))),
      # )

      if do_return_2d_tensor:
          attn_output = tf.reshape(attn_output, (batch_size*from_seq_length, self.embed_dim))
      else:
          attn_output = tf.reshape(attn_output, (batch_size,from_seq_length, self.embed_dim))

      context_layer=attn_output

      # # Take the dot product between "query" and "key" to get the raw
      # # attention scores.
      # #batch_size, num_attention_heads, from_seq_length, to_seq_length
      # # `attention_scores` = [B, N, F, T]
      # attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
      # attention_scores = tf.multiply(attention_scores,
      #                                1.0 / math.sqrt(float(size_per_head)))
      #
      # if attention_mask is not None:
      #   # `attention_mask` = [B, 1, F, T]
      #   attention_mask = tf.expand_dims(attention_mask, axis=[1])
      #
      #   # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      #   # masked positions, this operation will create a tensor which is 0.0 for
      #   # positions we want to attend and -10000.0 for masked positions.
      #   adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0
      #
      #   # Since we are adding it to the raw scores before the softmax, this is
      #   # effectively the same as removing these entirely.
      #   attention_scores += adder
      #
      # # Normalize the attention scores to probabilities.
      # # `attention_probs` = [B, N, F, T]
      # attention_probs = tf.nn.softmax(attention_scores)
      #
      # # This is actually dropping out entire tokens to attend to, which might
      # # seem a bit unusual, but is taken from the original Transformer paper.
      # attention_probs = dropout(attention_probs, attention_probs_dropout_prob)
      #
      # # `value_layer` = [B, T, N, H]
      # value_layer = tf.reshape(
      #     value_layer,
      #     [batch_size, to_seq_length, num_attention_heads, size_per_head])
      #
      # # `value_layer` = [B, N, T, H]
      # value_layer = tf.transpose(value_layer, [0, 2, 1, 3])
      #
      # # `context_layer` = [B, N, F, H]
      # context_layer = tf.matmul(attention_probs, value_layer)
      #
      # # `context_layer` = [B, F, N, H]
      # context_layer = tf.transpose(context_layer, [0, 2, 1, 3])
      #
      # if do_return_2d_tensor:
      #   # `context_layer` = [B*F, N*H]
      #   context_layer = tf.reshape(
      #       context_layer,
      #       [batch_size * from_seq_length, num_attention_heads * size_per_head])
      # else:
      #   # `context_layer` = [B, F, N*H]
      #   context_layer = tf.reshape(
      #       context_layer,
      #       [batch_size, from_seq_length, num_attention_heads * size_per_head])

      return context_layer



def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output = reshape_to_matrix(input_tensor)

  all_layer_outputs = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output

      with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
          attention_head = attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length)
          attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output

def longformer_model(input_tensor,
                      input_mask=None,
                      global_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,
                      is_training=False):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    input_mask: (optional) int32 Tensor of shape [batch_size, seq_length], with 1 for positions
    that can be attended to and 0 inpositions that should not be.
    global_mask: (optional) int32 Tensor of shape [batch_size, seq_length], with 1 for positions
    that can be attended to and 0 inpositions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """

  def _merge_to_attention_mask(attention_mask: tf.Tensor, global_attention_mask: tf.Tensor):
      # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
      # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
      # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
      if attention_mask is not None:
          attention_mask = attention_mask * (global_attention_mask + 1)
      else:
          # simply use `global_attention_mask` as `attention_mask`
          # if no `attention_mask` is given
          attention_mask = global_attention_mask + 1

      return attention_mask

  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output = reshape_to_matrix(input_tensor)

  if input_mask is None:
      input_mask = tf.fill([batch_size,seq_length], 1)

  attention_mask=input_mask

  if global_mask is not None:
      attention_mask= _merge_to_attention_mask(
          attention_mask, global_mask
      )
  #no self._pad_to_window_size

  # is index masked or global attention
  is_index_masked = tf.math.less(attention_mask, 1)
  is_index_global_attn = tf.math.greater(attention_mask, 1)
  is_global_attn = tf.math.reduce_any(is_index_global_attn)

  # We create a 3D attention mask from a 2D tensor mask.
  # Sizes are [batch_size, to_seq_length, 1, 1]
  # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
  # this attention mask is more simple than the triangular masking of causal attention
  # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
  attention_mask_shape = shape_list(attention_mask)
  extended_attention_mask = tf.reshape(
      attention_mask, (attention_mask_shape[0], attention_mask_shape[1], 1, 1)
  )
  extended_attention_mask = tf.cast(tf.math.abs(1 - extended_attention_mask), tf.dtypes.float32) * -10000.0

  all_layer_outputs = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output

      with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
          AttentionModule=LongformerAttentionModule(attention_window=8,layer_id=layer_idx)
          attention_head = AttentionModule.longformer_attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=extended_attention_mask,
              head_mask=None,
              is_index_masked=is_index_masked,
              is_index_global_attn=is_index_global_attn,
              is_global_attn=is_global_attn,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length,
              is_training=is_training)
          attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output

def longformer_go_model(input_tensor,
                      input_mask=None,
                      global_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,
                      is_training=False):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    input_mask: (optional) int32 Tensor of shape [batch_size, seq_length], with 1 for positions
    that can be attended to and 0 inpositions that should not be.
    global_mask: (optional) int32 Tensor of shape [batch_size, seq_length], with 1 for positions
    that can be attended to and 0 inpositions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """

  def _merge_to_attention_mask(attention_mask: tf.Tensor, global_attention_mask: tf.Tensor):
      # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
      # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
      # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
      if attention_mask is not None:
          attention_mask = attention_mask * (global_attention_mask + 1)
      else:
          # simply use `global_attention_mask` as `attention_mask`
          # if no `attention_mask` is given
          attention_mask = global_attention_mask + 1

      return attention_mask

  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output = reshape_to_matrix(input_tensor)

  if input_mask is None:
      input_mask = tf.fill([batch_size,seq_length], 1)

  attention_mask=input_mask

  if global_mask is not None:
      attention_mask= _merge_to_attention_mask(
          attention_mask, global_mask
      )
  #no self._pad_to_window_size

  # is index masked or global attention
  is_index_masked = tf.math.less(attention_mask, 1)
  is_index_global_attn = tf.math.greater(attention_mask, 1)
  is_global_attn = tf.math.reduce_any(is_index_global_attn)

  # We create a 3D attention mask from a 2D tensor mask.
  # Sizes are [batch_size, to_seq_length, 1, 1]
  # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
  # this attention mask is more simple than the triangular masking of causal attention
  # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
  attention_mask_shape = shape_list(attention_mask)
  extended_attention_mask = tf.reshape(
      attention_mask, (attention_mask_shape[0], attention_mask_shape[1], 1, 1)
  )
  extended_attention_mask = tf.cast(tf.math.abs(1 - extended_attention_mask), tf.dtypes.float32) * -10000.0

  all_layer_outputs = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output

      with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
          AttentionModule=LongformerAttentionModule(attention_window=8,layer_id=layer_idx)
          attention_head = AttentionModule.longformer_global_overwrite_attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=extended_attention_mask,
              head_mask=None,
              is_index_masked=is_index_masked,
              is_index_global_attn=is_index_global_attn,
              is_global_attn=is_global_attn,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length,
              is_training=is_training)
          attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output


def longformer_slide_model(input_tensor,
                      input_mask=None,
                      # global_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,
                      is_training=False):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    input_mask: (optional) int32 Tensor of shape [batch_size, seq_length], with 1 for positions
    that can be attended to and 0 inpositions that should not be.
    global_mask: (optional) int32 Tensor of shape [batch_size, seq_length], with 1 for positions
    that can be attended to and 0 inpositions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """

  def _merge_to_attention_mask(attention_mask: tf.Tensor, global_attention_mask: tf.Tensor):
      # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
      # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
      # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
      if attention_mask is not None:
          attention_mask = attention_mask * (global_attention_mask + 1)
      else:
          # simply use `global_attention_mask` as `attention_mask`
          # if no `attention_mask` is given
          attention_mask = global_attention_mask + 1

      return attention_mask

  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output = reshape_to_matrix(input_tensor)

  if input_mask is None:
      input_mask = tf.fill([batch_size,seq_length], 1)

  attention_mask=input_mask

  # if global_mask is not None:
  #     attention_mask= _merge_to_attention_mask(
  #         attention_mask, global_mask
  #     )
  #no self._pad_to_window_size

  # is index masked or global attention
  is_index_masked = tf.math.less(attention_mask, 1)
  # is_index_global_attn = tf.math.greater(attention_mask, 1)
  # is_global_attn = tf.math.reduce_any(is_index_global_attn)

  # We create a 3D attention mask from a 2D tensor mask.
  # Sizes are [batch_size, to_seq_length, 1, 1]
  # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
  # this attention mask is more simple than the triangular masking of causal attention
  # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
  attention_mask_shape = shape_list(attention_mask)
  extended_attention_mask = tf.reshape(
      attention_mask, (attention_mask_shape[0], attention_mask_shape[1], 1, 1)
  )
  extended_attention_mask = tf.cast(tf.math.abs(1 - extended_attention_mask), tf.dtypes.float32) * -10000.0

  all_layer_outputs = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output

      with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
          AttentionModule=LongformerAttentionModule(attention_window=8,layer_id=layer_idx)
          attention_head = AttentionModule.longformer_slid_attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=extended_attention_mask,
              head_mask=None,
              is_index_masked=is_index_masked,
              # is_index_global_attn=is_index_global_attn,
              # is_global_attn=is_global_attn,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length,
              is_training=is_training)
          attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output

def performer_model(input_tensor,
                      input_mask=None,
                      # global_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,
                      is_training=False):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    input_mask: (optional) int32 Tensor of shape [batch_size, seq_length], with 1 for positions
    that can be attended to and 0 inpositions that should not be.
    global_mask: (optional) int32 Tensor of shape [batch_size, seq_length], with 1 for positions
    that can be attended to and 0 inpositions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """

  def _merge_to_attention_mask(attention_mask: tf.Tensor, global_attention_mask: tf.Tensor):
      # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
      # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
      # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
      if attention_mask is not None:
          attention_mask = attention_mask * (global_attention_mask + 1)
      else:
          # simply use `global_attention_mask` as `attention_mask`
          # if no `attention_mask` is given
          attention_mask = global_attention_mask + 1

      return attention_mask

  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.

  # prev_output = reshape_to_matrix(input_tensor)
  prev_output = input_tensor

  # if input_mask is None:
  #     input_mask = tf.fill([batch_size,seq_length], 1)
  #
  # attention_mask=input_mask

  # if global_mask is not None:
  #     attention_mask= _merge_to_attention_mask(
  #         attention_mask, global_mask
  #     )
  #no self._pad_to_window_size

  # is index masked or global attention
  # is_index_masked = tf.math.less(attention_mask, 1)
  # is_index_global_attn = tf.math.greater(attention_mask, 1)
  # is_global_attn = tf.math.reduce_any(is_index_global_attn)

  # We create a 3D attention mask from a 2D tensor mask.
  # Sizes are [batch_size, to_seq_length, 1, 1]
  # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
  # this attention mask is more simple than the triangular masking of causal attention
  # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
  # attention_mask_shape = shape_list(attention_mask)
  # extended_attention_mask = tf.reshape(
  #     attention_mask, (attention_mask_shape[0], attention_mask_shape[1], 1, 1)
  # )
  # extended_attention_mask = tf.cast(tf.math.abs(1 - extended_attention_mask), tf.dtypes.float32) * -10000.0

  all_layer_outputs = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output
      layer_input=tf.reshape(layer_input,[-1,seq_length,hidden_size])

      with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
          attention_layer = fast_attention.SelfAttention(hidden_size, num_attention_heads, hidden_dropout_prob)
          bias = tf.ones([1])
          attention_head=attention_layer(layer_input,bias,training=is_training)
          # AttentionModule=LongformerAttentionModule(attention_window=8,layer_id=layer_idx)
          # attention_head = AttentionModule.longformer_slid_attention_layer(
          #     from_tensor=layer_input,
          #     to_tensor=layer_input,
          #     attention_mask=extended_attention_mask,
          #     head_mask=None,
          #     is_index_masked=is_index_masked,
          #     # is_index_global_attn=is_index_global_attn,
          #     # is_global_attn=is_global_attn,
          #     num_attention_heads=num_attention_heads,
          #     size_per_head=attention_head_size,
          #     attention_probs_dropout_prob=attention_probs_dropout_prob,
          #     initializer_range=initializer_range,
          #     do_return_2d_tensor=True,
          #     batch_size=batch_size,
          #     from_seq_length=seq_length,
          #     to_seq_length=seq_length,
          #     is_training=is_training)
          attention_head=reshape_to_matrix(attention_head)
          attention_heads.append(attention_head)

        attention_output = None
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          attention_output = dropout(attention_output, hidden_dropout_prob)
          # attention_output = layer_norm(attention_output + layer_input)
          layer_input_2d = tf.reshape(layer_input,[-1,hidden_size])
          attention_output = layer_norm(attention_output + layer_input_2d)

      # The activation is only applied to the "intermediate" hidden layer.
      with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # Down-project back to `hidden_size` then add the residual.
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output

def lstm_model(input_tensor,
                      input_mask=None,
                      # global_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False,
                      is_training=False):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    input_mask: (optional) int32 Tensor of shape [batch_size, seq_length], with 1 for positions
    that can be attended to and 0 inpositions that should not be.
    global_mask: (optional) int32 Tensor of shape [batch_size, seq_length], with 1 for positions
    that can be attended to and 0 inpositions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """


  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))


  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))
  # input_tensor=tf.reverse(input_tensor,axis=1)

  # 进行的多层双向神经网络
  fw_lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size) for i in range(num_hidden_layers)]
  bw_lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_size) for i in range(num_hidden_layers)]

  stack_lstm_fw = tf.nn.rnn_cell.MultiRNNCell(cells=fw_lstm_cells)
  stack_lstm_bw = tf.nn.rnn_cell.MultiRNNCell(cells=bw_lstm_cells)

  #[batch][time][input_width]
  outputs, _, = tf.nn.bidirectional_dynamic_rnn(cell_fw=stack_lstm_fw, cell_bw=stack_lstm_bw, inputs=input_tensor,
                                               dtype=tf.float32)
  outputs=tf.concat(outputs, 2)
  #[batch][time][cell_fw.output_size + cell_bw.output_size]


  if do_return_all_layers:
    final_outputs = [outputs]
    return final_outputs
  else:
    # final_output = reshape_from_matrix(prev_output, input_shape)
    final_output = outputs
    return final_output



def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
