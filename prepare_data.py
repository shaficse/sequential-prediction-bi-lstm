from collections import Counter
import tensorflow as tf
import sys
import os
import random

def create_vocabulary(corpus, vocab_path, max_freq, min_freq):
    counter = Counter()
    for c in corpus:
        counter.update(c.split(' '))
    with open(vocab_path, 'w') as of:
        of.write('<UKN>' + '\n')
        for w,f in counter.most_common():
            if f <= max_freq and f >= min_freq:
                of.write(w+'\n')

def load_vocabulary(vocab_path):
    vocabulary = {}
    with open(vocab_path, 'r') as f:
        for i,l in enumerate(f.readlines()):
            vocabulary[l.rstrip('\n')] = i
    return vocabulary

def transform_sentence(sentence, tags, vocabulary, max_sentence_length):
    sen_ids = []
    labels = []
    masks = []
    for w,t in zip(sentence.split(' '), tags.split(' ')):
        if w in vocabulary.keys():
            sen_ids.append(vocabulary[w])
            labels.append(int(t))
        else:
            sen_ids.append(0)
            labels.append(0)
    assert len(sen_ids) == len(labels)
    length = len(sen_ids)
    if length <= max_sentence_length:
        sen_ids.extend([0]*(max_sentence_length - length))
        labels.extend([0]*(max_sentence_length - length))
        masks = [1]*length
        masks.extend([0]*(max_sentence_length - length))
    else:
        sen_ids = sen_ids[0:max_sentence_length]
        labels = labels[0:max_sentence_length]
        masks = [1]*max_sentence_length
        length = max_sentence_length
    return sen_ids,labels, masks, length

def _int64_feature(value):
    # parameter value is a list
    return tf.train.Feature(int64_list = tf.train.Int64List(value = value))

def create_example(word_ids, word_tags, masks,length):
    features = {'sentence':_int64_feature(word_ids),
                'labels':_int64_feature(word_tags),
                'mask':_int64_feature(masks),
                'length':_int64_feature([length])}
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example

def create_datasets(raw_data_path, vocabulary,out_dir,max_sentence_length):
    examples = []
    with open(raw_data_path, 'r') as f:
        for l in f.readlines():
            sentence = l.split('\t')[0]
            tag = l.rstrip('\n').split('\t')[1]
            ids, label, mask,length = transform_sentence(sentence,tag,vocabulary,max_sentence_length)
            example = create_example(ids,label,mask,length)
            examples.append(example)

    random.shuffle(examples)
    train_file_path = os.path.join(os.path.abspath(out_dir),'train.tfrecords')
    valid_file_path = os.path.join(os.path.abspath(out_dir), 'validation.tfrecords')
    test_file_path = os.path.join(os.path.abspath(out_dir), 'test.tfrecords')

    writer = tf.python_io.TFRecordWriter(train_file_path)
    for e in examples[0:int(len(examples)*0.8)]:
        writer.write(e.SerializeToString())
    writer = tf.python_io.TFRecordWriter(valid_file_path)
    for e in examples[int(len(examples)*0.8):int(len(examples)*0.9)]:
        writer.write(e.SerializeToString())
    writer = tf.python_io.TFRecordWriter(test_file_path)
    for e in examples[int(len(examples)*0.9):]:
        writer.write(e.SerializeToString())

if __name__ == '__main__':
    corpus = []
    with open('data/dataset.txt', 'r') as f:
        corpus = [l.split('\t')[0] for l in f.readlines()]
    if len(sys.argv) > 1:
        MAX_FREQ = sys.argv[1]
    else:
        MAX_FREQ = 1000000

    if len(sys.argv) > 2:
        MIN_FREQ = sys.argv[2]
    else:
        MIN_FREQ = 3
    create_vocabulary(corpus,'data/vocabulary.txt',MAX_FREQ, MIN_FREQ)
    vocabulary = load_vocabulary('data/vocabulary.txt')

    if len(sys.argv) > 3:
        MAX_SEN_LENGTH = sys.argv[3]
    else:
        MAX_SEN_LENGTH = 70
    create_datasets('data/dataset.txt',vocabulary,'./data', max_sentence_length=MAX_SEN_LENGTH)