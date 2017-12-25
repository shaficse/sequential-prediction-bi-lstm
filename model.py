import tensorflow as tf
import bi_direc_rnn
import metrics

def create_train_op(lr,loss):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss,global_step=tf.train.get_global_step())
    return train_op

def create_model_fn():
    def model_fn(features,labels,mode,params):

        if mode == tf.estimator.ModeKeys.TRAIN:
            sentences = features['sentence']
            mask = features['mask']
            length = features['length']
            predictions,loss = bi_direc_rnn.model_impl(sentences,mask,length,labels,params,params.batch_size)
            tf.summary.scalar(name='train_loss',tensor=loss)
            train_op = create_train_op(params.learning_rate,loss)
            return tf.estimator.EstimatorSpec(loss=loss,train_op=train_op,mode=mode)
        elif mode == tf.estimator.ModeKeys.EVAL:
            sentences = features['sentence']
            mask = features['mask']
            length = features['length']
            predictions, loss = bi_direc_rnn.model_impl(sentences,mask,length,labels,params,params.eval_batch_size)
            eval_metric_ops = {}
            eval_metric_ops['precision'] = metrics.create_precesion_metric(predictions,labels,mask)
            eval_metric_ops['recall'] = metrics.create_recall_metric(predictions,labels,mask)
            return tf.estimator.EstimatorSpec(mode=mode,loss=loss,eval_metric_ops=eval_metric_ops)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = None
            return tf.estimator.EstimatorSpec(predictions=predictions)
    return model_fn