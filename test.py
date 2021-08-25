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

    '''
    Evaluate the best model (saved on disk) and report metrics
    on test data
    '''
    # qd.evaluate_model()

    # ''' test sample questions '''
    # q1 = 'What is the weather like?'
    # q2 = 'Is it hot today?'
    # print qd.are_duplicates(q1, q2)

