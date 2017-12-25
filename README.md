# sequential-prediction-bi-lstm
A bi-directional LSTM network for sequential prediction

It achieves the first subtask "Aspect Term Extraction" of SemEval-2014 task 4 "Aspect Based Sentiment Analysis (ABSA)". I just include one of the datasets provided by official website: Resturant Train Data. I separated it into train, validation and test dataset.

After some necessary preprocess, each word in the input sentence is converted to its corresponding word-embedding vector (I use the SENNA Embeddings http://ronan.collobert.com/senna/). Then each word vector is concantenated with its neighboring word vectors (the window size is a hyper-parameter) to make a "context vector" which capture the local information for each word. 

A bi-directional LSTM neural network is used to encode the whole context vector lists. After each time step, a fixed length hidden state is produced by network which representation the information for a word. This fixed length hidden state is feed into a fully connected layer and output a probability of how likely it is aspect term.

I implement the network using Tensorflow Estimator framework.
