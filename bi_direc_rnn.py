import tensorflow as tf
import helper

rnn_cells = {'BasicRNN':tf.nn.rnn_cell.BasicRNNCell, 'BasicLSTM':tf.nn.rnn_cell.BasicLSTMCell}

def load_word_embedding(vocab_path, word_embed_path):
    vocabulary, vocab_dict = helper.load_vocab(vocab_path)
    glove_vectors, glove_dict = helper.load_glove_vectors(word_embed_path,vocabulary)
    vocab_size = len(vocabulary)
    word_dim = glove_vectors.shape[1]
    embedding_matrix = helper.build_initial_embedding_matrix(vocab_dict=vocab_dict,glove_vectors=glove_vectors,glove_dict=glove_dict,embedding_dim=word_dim)
    embedding_W = tf.get_variable('word_embedding_W',dtype=tf.float32,initializer=embedding_matrix,trainable=False)
    return embedding_W

def model_impl(sentence,mask,length,labels,hparam,batch_size):
    #convert to word vector
    with tf.name_scope(name='embedding_lookup') as ns:
        embedding_W = load_word_embedding(hparam.vocab_path,hparam.glove_path)
        sentence = tf.nn.embedding_lookup(embedding_W,sentence)

    # make context vector
    with tf.name_scope(name='local_context_maker') as ns:
        padding_size = int((hparam.context_width)/2)
        padding = tf.constant(value=[[0,0],[padding_size,padding_size],[0,0]],name='padding')
        sentence_pad = tf.pad(sentence,padding,name='paded_sentence') #batch, max_length+context_width - 1, word_dim
        sentence_slices = []
        for i in range(hparam.max_sentence_length):
            begin = [0,i,0]
            slice = tf.slice(sentence_pad,begin=begin,size=[batch_size,hparam.context_width,hparam.word_dim])
            slice = tf.expand_dims(slice,axis=1)
            sentence_slices.append(slice)
        sentence_concat = tf.concat(sentence_slices,axis=1) #batch, max_sen_length, context_width, word_dim
        sentence_split = tf.split(sentence_concat,hparam.context_width,axis=2)
        sentence_split = [tf.squeeze(s,axis=2) for s in sentence_split]
        sentence = tf.concat(sentence_split,axis=2) #batch, max_sen_length, context_width*word_dim

    # encode use bi-LSTM
    with tf.variable_scope('rnn_encoding') as ns:
        with tf.variable_scope('forward') as vs:
            cell = rnn_cells[hparam.rnn_cell_type](num_units=hparam.rnn_num_units)
            state = cell.zero_state(batch_size,dtype=tf.float32)
            rnn_inputs = tf.split(sentence,hparam.max_sentence_length,axis=1)
            rnn_inputs = [tf.squeeze(input,1) for input in rnn_inputs]

            rnn_states_fw = [] #batch_size, rnn_state_size, length = max_sen_length
            for rnn_input in rnn_inputs:
                _,state = cell(rnn_input,state)
                if isinstance(state,tuple):
                    rnn_states_fw.append(state.h)
                else:
                    rnn_states_fw.append(state)

        with tf.variable_scope('backward') as vs:
            length = tf.squeeze(length,axis=1,name='squeeze_length_to_dim1')
            cell = rnn_cells[hparam.rnn_cell_type](num_units=hparam.rnn_num_units)
            state = cell.zero_state(batch_size, dtype=tf.float32)
            rnn_inputs_reverse = tf.reverse_sequence(sentence,length,seq_dim=1,batch_dim=0) #batch, max_sen_length, context_width*word_dim
            rnn_inputs_reverse = tf.split(rnn_inputs_reverse, hparam.max_sentence_length, axis=1)
            rnn_inputs_reverse = [tf.squeeze(input, 1) for input in rnn_inputs_reverse]

            rnn_states_bw = [] #batch_size, context_width*word_dim
            for rnn_input in rnn_inputs_reverse:
                _, state = cell(rnn_input, state)
                if isinstance(state, tuple):
                    rnn_states_bw.append(tf.expand_dims(state.h,axis=1))
                else:
                    rnn_states_bw.append(tf.expand_dims(state, axis=1))

            rnn_states_bw = tf.concat(values=rnn_states_bw, axis=1)  # [batch_size, max_sen_length, state_size]
            rnn_states_bw = tf.reverse_sequence(input=rnn_states_bw, seq_lengths=length, seq_axis=1, batch_axis=0,
                                                 name='bw_reverse_back')  # [batch_size, max_sen_length, state_size]
            rnn_states_bw = tf.split(value=rnn_states_bw, num_or_size_splits=hparam.max_sentence_length, axis=1,
                                      name='bw_reverse_back_split')  # [batch_size, 1, state_size]
            rnn_states_bw = [tf.squeeze(bs, axis=1, name='bw_reverse_back_squeeze') for bs in
                             rnn_states_bw] # [batch_size, state_size]
    # softmax
    with tf.variable_scope('softmax_and_loss'):
        if isinstance(cell.state_size, tuple):
            W = tf.get_variable('W',dtype=tf.float32,initializer=tf.random_normal(shape=[cell.state_size[0]*2,1]))
        else:
            W = tf.get_variable('W', shape=[cell.state_size * 2, 1], dtype=tf.float32,
                                initializer=tf.random_normal(shape=[cell.state_size * 2, 1]))
        b = tf.get_variable('b',dtype=tf.float32, initializer=tf.zeros(shape=[1]))

        rnn_states = []
        for i in range(hparam.max_sentence_length):
            rnn_states.append(tf.concat([rnn_states_fw[i],rnn_states_bw[i]],1))

        predictions = []
        logits = []
        for i in range(hparam.max_sentence_length):
            a = rnn_states[i]
            y = tf.matmul(a,W) + b #batch, 1
            prediction = tf.sigmoid(y) #batch, 1
            predictions.append(prediction) #batch, 1
            logits.append(y) #batch, 1

        predictions = tf.concat(predictions,axis=1) #batch_size, max_sen_length
        predictions = tf.round(predictions,name='final_predictions')
        #predictions = tf.multiply(predictions,mask)

        logits = tf.concat(logits,axis=1) #batch_size, max_sen_length
        logits = tf.multiply(logits,tf.to_float(mask),name='final_logits')
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.to_float(labels),logits=logits) #batch_size, max_sen_length

        loss = tf.reduce_mean(cross_entropy,name='final_loss')

    return predictions, loss