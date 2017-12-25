import tensorflow as tf
from collections import namedtuple
import os

# Model Parameters
tf.flags.DEFINE_integer("word_dim", 50, "Dimensionality of the embeddings")
tf.flags.DEFINE_integer('rnn_num_units', 256, 'Num of rnn cells')
tf.flags.DEFINE_float('keep_prob', 1.0, 'the keep prob of rnn state')
tf.flags.DEFINE_string('rnn_cell_type', 'BasicLSTM', 'the cell type in rnn')
tf.flags.DEFINE_integer('context_width',3,'local context width')

# Pre-trained embeddings
tf.flags.DEFINE_string("glove_path", './data/word_embed_50.txt', "Path to pre-trained Glove vectors")
tf.flags.DEFINE_string("vocab_path", './data/vocabulary.txt', "Path to vocabulary.txt file")
tf.flags.DEFINE_integer('max_sentence_length', 70,'the max sentence length')

# Training Parameters, 2435 train 304 valid 305 test
tf.flags.DEFINE_integer("batch_size", 5, "Batch size during training")
tf.flags.DEFINE_integer("eval_batch_size", 304, "Batch size during evaluation")
tf.flags.DEFINE_integer('num_epochs', 10, 'the number of epochs')
tf.flags.DEFINE_integer('eval_step', 487, 'eval every n steps')
tf.flags.DEFINE_boolean('shuffle_batch',False, 'whether shuffle the train examples when batch')
tf.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate")

FLAGS = tf.flags.FLAGS

HParams = namedtuple(
  "HParams",
  [ "eval_step",
    "batch_size",
    "word_dim",
    "eval_batch_size",
    "learning_rate",
    "glove_path",
    "vocab_path",
    "num_epochs",
    'rnn_num_units',
    'keep_prob',
    'rnn_cell_type',
    'max_sentence_length',
    'context_width',
    'shuffle_batch'
  ])

def create_hparam():
  return HParams(
    eval_step = FLAGS.eval_step,
    batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    learning_rate=FLAGS.learning_rate,
    word_dim=FLAGS.word_dim,
    glove_path=FLAGS.glove_path,
    vocab_path=FLAGS.vocab_path,
    num_epochs=FLAGS.num_epochs,
    rnn_num_units=FLAGS.rnn_num_units,
    keep_prob=FLAGS.keep_prob,
    rnn_cell_type=FLAGS.rnn_cell_type,
    max_sentence_length=FLAGS.max_sentence_length,
    context_width = FLAGS.context_width,
    shuffle_batch = FLAGS.shuffle_batch
  )

def write_hparams_to_file(hp, model_dir):
  with open(os.path.join(os.path.abspath(model_dir),'hyper_parameters.txt'), 'w') as f:
    f.write('batch_size: {}\n'.format(hp.batch_size))
    f.write('learning_rate: {}\n'.format(hp.learning_rate))
    f.write('num_epochs: {}\n'.format(hp.num_epochs))
    f.write('rnn_num_units: {}\n'.format(hp.rnn_num_units))
    f.write('keep_prob: {}\n'.format(hp.keep_prob))
    f.write('rnn_cell_type: {}\n'.format(hp.rnn_cell_type))
    f.write('context_width {}\n'.format(hp.context_width))
