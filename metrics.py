import tensorflow as tf

def create_precesion_metric(predictions,labels,masks):
    return tf.metrics.precision(labels=labels,predictions=predictions,weights=masks)

def create_recall_metric(predictions,labels,masks):
    return tf.metrics.recall(labels=labels,predictions=predictions,weights=masks)
