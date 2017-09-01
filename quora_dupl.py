# -*- coding: utf-8 -*-

import os
import sys
import zipfile
import time

import six
from six.moves.urllib import request

import numpy as np
import pandas as pd

from collections import defaultdict
from unidecode import unidecode

import tensorflow as tf

import keras
import keras.backend as K

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

from keras.layers import Input, Dense, TimeDistributed, merge, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Lambda

from keras.models import Model

from keras import regularizers
from keras.regularizers import l2

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, CSVLogger, ModelCheckpoint

from keras.backend.tensorflow_backend import set_session

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, f1_score


GLOVE_URL   = 'http://nlp.stanford.edu/data'
GLOVE_DATA  = 'glove.840B.300d.zip'
QUORA_DATA  = 'quora_duplicate_questions.tsv'
QUORA_URL   = 'http://qim.ec.quoracdn.net/'


def runtime(func):
    '''
    decorator for runtime evaluation
    '''
    def myfunc(*args, **kwargs):
        ### time.clock()'s precision higher than time.time()
        t_start = time.clock()
        retval = func(*args, **kwargs)
        t_end = time.clock()
        print '{} runtime: {}'.format(func.__name__, t_end - t_start)
        return retval
    return myfunc


class DecomposeAttention(object):

    def __init__(self, model_type ='dec_attn', dataset = 'quora'):

        ''' model and training parameters '''

        self.__embedding_size = 300             ### embedding size from GloVe
        self.__sentence_max_len = 72            ### reasonable max sentence length from Quora dataset = 72.
                                                ### median q1, q2 length = 10
        self.__vocab_size = 0                   ### Size of vocabulary

        self.__batch_size = 128                 ### Batch size value = 32 from Parikh et al. Can try different values
        self.__max_epochs = 50                  ### max num epochs = 50. This is OK as we are doing EarlyStopping

        self.__dropout_rate = 0.2               ### Dropout rate from Parikh et al. (other values = 0.25, 0.4, 0.5)
        self.__l2_reg_val = 1e-5                ### L2 regularization strength
        self.__activation_func = 'relu'         ### Parikh et al. used ReLU activation on FC layers
        self.__sgd_optimizer = 'Adam'           ### SDG optimizer type. Parikh et al. used Adagrad on SNLI
        self.__patience = 8                     ### for early stopping

        self.__model_type = model_type          ### Name of model type
        self.__dataset = dataset                ### Name of dataset

        self.__num_train = 300000               ### Number of training examples to use - as directed, for this assignment
        self.__num_test = 0                     ### Number of test examples = dataset size - self.__num_train; set it later

        ''' model and embeddings '''
        self.model = None
        self.glove_embedding = defaultdict(np.array)

        self.question_proc = None
        self.question_embedding = None

        ''' data containers '''
        self.train = []
        self.test = []

        ''' miscellaneous options '''
        self.log_tensorflow_run = False


    def download_data(self):

        '''
        download_data()

        run this first in case the dataset and GloVe embeddings are not available
        if they are available, copy them into $PWD/data
        '''

        if not os.path.exists('data'):
            os.mkdir('data')

        if not os.path.exists('data/glove.840B.300d.txt'):
            ''' get glove data '''
            print 'Downloading {:s}'.format(GLOVE_DATA)
            request.urlretrieve('{:s}/{:s}'.format(GLOVE_URL, GLOVE_DATA))
            print 'Done'

            print 'Processing {:s}'.format(GLOVE_DATA)
            with zipfile.ZipFile('data/glove.840B.300d.zip') as gfile:
                gfile.extractall('data')

            print 'Removing {:s}'.format(GLOVE_DATA)
            os.remove('data/' + GLOVE_DATA)

        if not os.path.exists('data/quora_duplicate_questions.tsv'):
            ''' get quora dataset and prep '''
            print 'Downloading {:s}...'.format(QUORA_DATA)
            request.urlretrieve('{:s}/{:s}'.format(QUORA_URL, QUORA_DATA))
            print 'Done'


    def load_data(self):

        '''
        load_data()

        called by process_data
        this routine imports Quora tsv into Pandas and returns train/test splits after Unicode cleanup

        '''

        df = pd.read_csv('./data/quora_duplicate_questions.tsv', delimiter='\t', dtype={'is_duplicate': 'int'})

        df['question1'] = df['question1'].apply(lambda x: unicode(str(x), 'utf-8'))
        df['question2'] = df['question2'].apply(lambda x: unicode(str(x), 'utf-8'))

        ### fix for https://github.com/fchollet/keras/issues/1072
        df['question1'] = df['question1'].apply(lambda x: unidecode(x))
        df['question2'] = df['question2'].apply(lambda x: unidecode(x))

        self.__num_test = df.shape[0] - self.__num_train
        print 'Number of training pairs: {}'.format(self.__num_train)
        print 'Number of testing pairs:  {}'.format(self.__num_test)

        df = df.drop(['id', 'qid1', 'qid2'], axis=1)

        train_df = df[:self.__num_train]
        test_df = df[self.__num_train:]

        train, test = [], []

        train.append(list(train_df.question1))
        train.append(list(train_df.question2))
        train_categories = np_utils.to_categorical(train_df.is_duplicate, 2)
        train.append(train_categories)

        test.append(list(test_df.question1))
        test.append(list(test_df.question2))
        test_categories = np_utils.to_categorical(test_df.is_duplicate.values, 2)
        test.append(test_categories)

        return train, test


    def process_data(self):

        '''
        process_data()

        parse Quora question data, cleanup and pad question sentences to prepare for embedding
        '''

        self.train, self.test = self.load_data()

        ### Using keras tokenizer ;; create a Tokenizer instance
        self.question_proc = Tokenizer(lower=False, filters='')

        ### Fit all items in question1 and question2 to the Tokenizer
        self.question_proc.fit_on_texts(self.train[0] + self.train[1])

        self.__vocab_size = len(self.question_proc.word_counts)

        ### adding 1 here as documentation says `0` is a reserved index not assigned to any word
        self.__vocab_size += 1

        ### Convert words in sentences to a sequence of ints and pad zeros
        def do_padding(x, __max_len):
            return pad_sequences(sequences=self.question_proc.texts_to_sequences(x), maxlen=__max_len)

        def pad_data(x):
            return do_padding(x[0], self.__sentence_max_len), do_padding(x[1], self.__sentence_max_len), x[2]

        ### For train and test data we return a tuple of size 3 corresponding to question1, question2 and ground-truth
        self.train = pad_data(self.train)
        self.test = pad_data(self.test)


    def process_glove(self):

        '''
        process_glove()

        process GloVe embeddings
        '''

        try:
            open('data/glove.840B.300d.txt', 'r')
        except IOError:
            print 'ERROR: Cannot find data/glove.840B.300d.txt\n\n\
            Copy the GloVe download file to data directory.\n\
            -or- run download_data() first.\n\n\
            Note: Downloading GloVe from nlp.stanford.edu \n\
            may take time depending on the speed of your internet connection'
        else:
            print 'Found GloVe file: data/glove.840B.300d.txt'
            print 'Starting processing of sentence embeddings...'

        ### create embedding matrix
        embed_index = {}
        for l in open('data/glove.840B.300d.txt', 'r'):
            value = l.split(' ')
            word = value[0]
            embed_index[word] = np.asarray(value[1:], dtype='float32')
            self.glove_embedding = embed_index[word]

        embed_matrix = np.zeros((self.__vocab_size, self.__embedding_size))

        unknown = []
        for word, i in self.question_proc.word_index.items():
            vec = embed_index.get(word)
            if vec is None:
                unknown.append(word)
            else:
                embed_matrix[i] = vec

        ### save the embedding matrix for future runs
        np.save('data/GloVe_' + self.__dataset + '.npy', embed_matrix)
        open('unknown_words.txt', 'w').write(str(unknown))


    def process_embedding(self):

        '''
        process_embedding()

        reads in processed embeddings and creates the embedding layer
        '''

        if not os.path.exists('data/GloVe_' + self.__dataset + '.npy'):
            self.process_glove()

        embed_matrix = np.load('data/GloVe_' + self.__dataset + '.npy')
        self.question_embedding = Embedding(input_dim = self.__vocab_size,
                                output_dim = self.__embedding_size,
                                input_length = self.__sentence_max_len,
                                trainable = False,
                                weights = [embed_matrix],
                                name = 'embed_quora')


    def gen_attn_model(self, test_mode = False):

        '''
        gen_attn_model()

        Create decomposable attention model along the lines described in:

        A Decomposable Attention Model for Natural Language Inference, Parikh et al.
        https://arxiv.org/abs/1606.01933

        '''


        ### https://www.tensorflow.org/tutorials/using_gpu
        config = tf.ConfigProto(log_device_placement=self.log_tensorflow_run)
        config.gpu_options.allow_growth = True

        ### Play with the following options in a multi-GPU environment
        #config.gpu_options.per_process_gpu_memory_fraction = 0.2

        set_session(tf.Session(config=config))


        ### Input Representation
        ### Layer definitions and network build
        question1 = Input(shape=(self.__sentence_max_len,), dtype='int32')
        question2 = Input(shape=(self.__sentence_max_len,), dtype='int32')
        q1_embed = self.question_embedding(question1)
        q2_embed = self.question_embedding(question2)

        ### Question embedding projection with TimeDistributed Dense (== FC layer from Parikh et al.)
        q_embed_project = TimeDistributed(Dense(200, activation='relu',
                                    kernel_regularizer=l2(self.__l2_reg_val),
                                    bias_regularizer=l2(self.__l2_reg_val)))

        q1_embed = Dropout(self.__dropout_rate)(q_embed_project(q1_embed))
        q2_embed = Dropout(self.__dropout_rate)(q_embed_project(q2_embed))

        ### Attend
        F_q1, F_q2 = q1_embed, q2_embed

        for i in range(2): # Applying Decomposable Score Function
            scoreF = TimeDistributed(Dense(200, activation='relu',
                                        kernel_regularizer=l2(self.__l2_reg_val),
                                        bias_regularizer=l2(self.__l2_reg_val)))

            F_q1 = Dropout(self.__dropout_rate)(scoreF(F_q1))
            F_q2 = Dropout(self.__dropout_rate)(scoreF(F_q2))

        ### Calculate e_ij (unnormalized attention scores) - dot product of F(a_i).T, F(b_i)
        Eq1q2 = keras.layers.Dot(axes=(2, 2))([F_q1, F_q2])

        ### Normalization of attention scores
        Eq1 = Lambda(lambda x: keras.activations.softmax(x))(Eq1q2)
        Eq2 = keras.layers.Permute((2, 1))(Eq1q2)
        Eq2 = Lambda(lambda x: keras.activations.softmax(x))(Eq2)

        ### Soft alignment prior to concatenation
        q1_align = keras.layers.Dot((2, 1))([Eq1, q1_embed])
        q2_align = keras.layers.Dot((2, 1))([Eq2, q2_embed])

        ### Concatenate and compare aligned phrases - with FC network G
        q1_align = keras.layers.concatenate([q1_embed, q1_align])
        q2_align = keras.layers.concatenate([q2_embed, q2_align])

        for i in range(2):
            functionG = TimeDistributed(Dense(200, activation='relu',
                                        kernel_regularizer=l2(self.__l2_reg_val),
                                        bias_regularizer=l2(self.__l2_reg_val)))

            q1_align = functionG(q1_align) # [batch_size, Psize, units]
            q2_align = functionG(q2_align) # [batch_size, Hsize, units]
            q1_align = Dropout(self.__dropout_rate)(q1_align)
            q2_align = Dropout(self.__dropout_rate)(q2_align)

        ### Aggregate each comparison vector
        aggregate_vec = Lambda(lambda X: K.reshape(K.sum(X, axis=1, keepdims=True), (-1, 200)))
        v_q1 = aggregate_vec(q1_align)
        v_q2 = aggregate_vec(q2_align)

        ### Form the final concatenation of aggregated v1 and v2 - that will be input to classifier H
        q1_q2_final = keras.layers.concatenate([v_q1, v_q2])
        for i in range(2):
            q1_q2_final = Dense(200, activation='relu',
                        kernel_regularizer=l2(self.__l2_reg_val),
                        bias_regularizer=l2(self.__l2_reg_val))(q1_q2_final)

            q1_q2_final = Dropout(self.__dropout_rate)(q1_q2_final)
            q1_q2_final = BatchNormalization()(q1_q2_final)

        ### Softmax out!
        q1_q2_final = Dense(2, activation='softmax')(q1_q2_final)
        self.model = Model(inputs=[question1, question2], outputs=q1_q2_final)


    def compile_model(self, use_existing_weights=False):

        '''
        Compile model

        if use_existing_weights is True, load any pre-existing weights file from previous runs
        '''

        self.model.compile(optimizer=self.__sgd_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

        output_folder = './data/'
        chkpt_file = output_folder + self.__model_type + '_best_' + self.__dataset + '.hdf5'

        if use_existing_weights:
            if os.path.exists(chkpt_file):
                print 'Attempting loading of weights from: {}'.format(chkpt_file)
                self.model.load_weights(chkpt_file, by_name=True)
                print 'Load weights successful'
            else:
                print 'Load weights UNSUCCESSFUL. Cannot find weights file: {}'.format(chkpt_file)


    def train_model(self, n_epochs=None):

        '''
        Train the model

        Callbacks: early stopping, learning rate reduction, model checkpointing and train logging
        '''

        output_folder = './data/'
        chkpt_file = output_folder + self.__model_type + '_best_' + self.__dataset + '.hdf5'

        t_cbks = [EarlyStopping(patience=self.__patience),
                ReduceLROnPlateau(patience=5, verbose=1),
                ModelCheckpoint(chkpt_file, verbose=True, save_best_only=True, save_weights_only=True),
                CSVLogger(filename=self.__model_type + '_' + self.__dataset + '_log.csv')]

        ### validation data is being set to test set (for now); to be improved
        self.model.fit(x = [self.train[0],self.train[1]], y = self.train[2],
                    batch_size = self.__batch_size,
                    epochs = n_epochs if n_epochs is not None else self.__max_epochs,
                    validation_data=([self.test[0], self.test[1]], self.test[2]),
                    callbacks = t_cbks)


    def evaluate_model(self, use_existing_weights=True):

        '''
        Evaluate the model

        By default, this will load the best checkpointed model saved to disk

        To evaluate the in-memory model performance, set 'use_existing_weights' to False and call this

        '''

        output_folder = './data/'
        chkpt_file = output_folder + self.__model_type + '_best_' + self.__dataset + '.hdf5'

        if use_existing_weights:
            if os.path.exists(chkpt_file):
                print 'Attempting loading of weights from: {}'.format(chkpt_file)
                self.model.load_weights(chkpt_file, by_name=True)
                print 'Load weights successful'
            else:
                print 'Load weights UNSUCCESSFUL. Cannot find weights file: {}'.format(chkpt_file)

        ### we can do prediction in batches -or- pass the entire testset at once (choice to be made on user need)
        predicted_softmax = self.model.predict(x = [self.test[0], self.test[1]])
        predicted_dups = np.argmax(predicted_softmax, axis=1)
        actual_dups = np.argmax(self.test[2], axis=1)

        ### Using scikit-learn's metrics
        true_neg, false_pos, false_neg, true_pos = confusion_matrix(actual_dups, predicted_dups).ravel()

        accuracy = accuracy_score(actual_dups, predicted_dups)

        precision = precision_score(actual_dups, predicted_dups)
        recall = precision_score(actual_dups, predicted_dups)
        f1score = f1_score(actual_dups, predicted_dups)
        auc = roc_auc_score(actual_dups, predicted_dups)

        print '\n'
        print '*'*50
        print ' '*2 + ' Prediction Metrics ' + ' '*29
        print '*'*50
        print '\t Accuracy  = {}'.format(accuracy)
        print '\n'
        print '\t Precision = {}'.format(precision)
        print '\t Recall    = {}'.format(recall)
        print '\t F1-Score  = {}'.format(f1score)
        print '\t AUC       = {}'.format(auc)
        print '\n'
        print '\t True  Pos = {}'.format(true_pos)
        print '\t False Pos = {}'.format(false_pos)
        print '\t False Neg = {}'.format(false_neg)
        print '\t True  Neg = {}'.format(true_neg)
        print '*'*50


    @runtime
    def are_duplicates(self, q1, q2):

        '''
        Predict if two questions q1 and q2 are duplicates

        Enables batch-mode/interactive prediction
        '''
        assert type(q1) is str
        assert type(q2) is str
        ### current supported question length <= 72 words/tokens
        assert(len(q1.split(' ')) <= 72 and len(q2.split(' ')) <= 72)

        print '\nChecking: '
        print 'q1: {}'.format(q1)
        print 'q2: {}'.format(q2)

        q_embed = lambda question: pad_sequences(sequences=self.question_proc.texts_to_sequences(question), maxlen=self.__sentence_max_len)

        ### TODO: check if the vocabulary/tokens in q1 and q2 have embeddings 

        q1_embed = q_embed(q1)
        q2_embed = q_embed(q2)

        dup_predict = self.model.predict(x=[q1_embed, q2_embed], batch_size=1, verbose=True)
        is_duplicate = np.argmax(dup_predict, axis=1)

        if is_duplicate:
            return 'yes'
        else:
            return 'no'











