import tensorflow as tf
import model
import hparam
import input
import time
import os
from tensorflow.python import debug as tf_debug

tf.flags.DEFINE_string('model_dir',None,'The model dir')
tf.flags.DEFINE_boolean('debug', False, 'debug mode')

FLAGS = tf.flags.FLAGS

if FLAGS.model_dir:
    MODEL_DIR = FLAGS.model_dir
else:
    TIMESTAMP = str(time.time())
    MODEL_DIR = os.path.join("./model", TIMESTAMP)

TRAIN_FILE = './data/train.tfrecords'
VALID_FILE = './data/validation.tfrecords'

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_arg):
    hyper_parameters = hparam.create_hparam()

    train_config = tf.contrib.learn.RunConfig(gpu_memory_fraction=1, save_summary_steps=hyper_parameters.eval_step,
                                              save_checkpoints_steps=hyper_parameters.eval_step, log_step_count_steps=1000)

    estimator = tf.estimator.Estimator(model_fn=model.create_model_fn(), model_dir=MODEL_DIR, config=train_config, params=hyper_parameters)

    monitors_list = []
    if FLAGS.debug:
        debuger = tf_debug.LocalCLIDebugHook()
        monitors_list.append(debuger)

    valid_input_fn = input.create_input_fn(tf.estimator.ModeKeys.EVAL,[VALID_FILE],hyper_parameters.eval_batch_size,1,False)
    valid_monitor = tf.contrib.learn.monitors.ValidationMonitor(input_fn=valid_input_fn,every_n_steps=hyper_parameters.eval_step)
    monitors_list.append(valid_monitor)

    hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks(monitors_list, estimator)
    train_input_fn = input.create_input_fn(tf.estimator.ModeKeys.TRAIN,[TRAIN_FILE],hyper_parameters.batch_size,hyper_parameters.num_epochs,hyper_parameters.shuffle_batch)
    estimator.train(input_fn=train_input_fn, hooks= hooks)

    hparam.write_hparams_to_file(hyper_parameters,MODEL_DIR)

if __name__ == '__main__':
    tf.app.run()