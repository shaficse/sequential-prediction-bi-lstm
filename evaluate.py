import tensorflow as tf
import model
import input
from tensorflow.python import debug as tf_debug
import hparam

tf.flags.DEFINE_string('model_dir', None, 'model dir')
tf.flags.DEFINE_boolean('debug', False, 'debug mode')
FLAGS = tf.flags.FLAGS

def main(unused_arg):
    eval_model_fn = model.create_model_fn()
    if not FLAGS.model_dir:
        raise KeyError()
    else:
        MODEL_DIR = FLAGS.model_dir

    hyper_parameters = hparam.create_hparam()

    estimator = tf.estimator.Estimator(eval_model_fn, model_dir=MODEL_DIR, params=hyper_parameters)

    EVAL_FILE = './data/validation.tfrecords'
    eval_input_fn = input.create_input_fn(tf.estimator.ModeKeys.EVAL,[EVAL_FILE],hyper_parameters.eval_batch_size,1,False)

    monitors_list = []
    if FLAGS.debug:
        debuger = tf_debug.LocalCLIDebugHook()
        monitors_list.append(debuger)
    hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks(monitors_list, estimator)

    eval_result = estimator.evaluate(input_fn=eval_input_fn,hooks=hooks)

    print('precision: {}'.format(eval_result['precision']))
    print('recall: {}'.format(eval_result['recall']))

if __name__ == '__main__':
    tf.app.run()