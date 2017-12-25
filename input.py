import tensorflow as tf

MAX_SEN_LENGTH = 50

def create_feature_spec():
    spec = {}
    spec['sentence'] = tf.FixedLenFeature(shape=[MAX_SEN_LENGTH],dtype=tf.int64)
    spec['length'] = tf.FixedLenFeature(shape=[1],dtype=tf.int64)
    spec['mask'] = tf.FixedLenFeature(shape=[MAX_SEN_LENGTH],dtype=tf.int64)
    spec['labels'] = tf.FixedLenFeature(shape=[MAX_SEN_LENGTH], dtype=tf.int64)
    return spec

def read_and_decode(filenames,num_of_epochs):
    filename_queue = tf.train.string_input_producer(filenames, num_of_epochs, shuffle=False)

    reader = tf.TFRecordReader("zhang_xiang_reader")
    _,serilized_example = reader.read(queue=filename_queue,name='read_into_data')

    feature_spec = create_feature_spec()
    example = tf.parse_single_example(serilized_example,features=feature_spec)

    return example


def create_input_fn(mode, tfrecord_files,batch_size, num_epochs,shuffle_batch):
    def input_fn():
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            with tf.name_scope('input_layer') as ns:
                single_example = read_and_decode(tfrecord_files,num_epochs)
                if shuffle_batch:
                    min_after_dequeue = 10000
                    capacity = min_after_dequeue + 3 * batch_size
                    batch_example_features = tf.train.shuffle_batch(single_example, batch_size,
                                                           min_after_dequeue=min_after_dequeue,
                                                           capacity=capacity)
                else:
                    batch_example_features = tf.train.batch(single_example,batch_size)
                batch_example_label = batch_example_features.pop('labels')
            return batch_example_features, batch_example_label
        else:
            features = None
            return features, None
    return input_fn