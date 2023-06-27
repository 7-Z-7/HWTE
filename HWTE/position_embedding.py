import tensorflow as tf

def positional_embedding(pos_seq, inv_freq, bsz=None):
    sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    if bsz is not None:
        return tf.tile(pos_emb[:, None, :], [1, bsz, 1])
    else:
        return pos_emb[:, None, :]

def sinusoidal_positional_embedding(max_position_embeddings,width):
    pos_seq = tf.range(max_position_embeddings - 1, -1, -1.0)
    inv_freq = [1 / (10000.0 ** (i / width)) for i in range(0, width, 2)]
    inv_freq = tf.constant(inv_freq)
    full_position_embeddings = positional_embedding(pos_seq, inv_freq)
    full_position_embeddings = tf.squeeze(full_position_embeddings)
    return full_position_embeddings

def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.truncated_normal_initializer(stddev=initializer_range)

def trainable_positional_embedding(max_position_embeddings,width,initializer_range=0.02,position_embedding_name='trainable_positional_embedding'):
    full_position_embeddings = tf.get_variable(
        name=position_embedding_name,
        shape=[max_position_embeddings, width],
        initializer=create_initializer(initializer_range))
    return full_position_embeddings


# position_embeddings = tf.reshape(position_embeddings,
#                                  position_broadcast_shape)
