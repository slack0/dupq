# -*- coding: utf-8 -*-

from quora_dupl import *

if __name__ == '__main__':

    ''' create an instance of DecomposeAttention class '''
    qd = DecomposeAttention()

    ''' Call process data for data load/cleanup '''
    qd.process_data()

    ''' Process GloVe embeddings '''
    qd.process_embedding()

    ''' Create model '''
    qd.gen_attn_model()

    ''' Compile/load previous model '''
    qd.compile_model(use_existing_weights=False)

    ''' 
    Train the model; specify n_epochs if desired 
    Max epochs set in the main class; early stopping enbled

    First 300K question pairs used for training
    '''
    qd.train_model(n_epochs=10)

    '''
    Evaluate the best model (saved on disk) and report metrics
    on test data
    '''
    qd.evaluate_model()
    
    ''' test sample questions '''
    q1 = 'What is the weather like?'
    q2 = 'Is it hot today?'
    print qd.are_duplicates(q1, q2)

