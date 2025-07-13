'''
@author: [Anonymized]
'''

import numpy as np
from scipy import sparse # optional -- set do_sparse_dot_prod=False in multitopic_coherence_dense() to avoid (but it usually is much faster w/sparse)


def coherence_dense(topic_word_vec, doc_term_matrix, num_topwords_to_check, eps_to_add=0.000000001):
    # UMass coherence computed using a (dense) np.array doc_term_matrix
    doc_term_matrix = doc_term_matrix.astype(np.bool)
    topic_coherence_score = 0.0
    topic_rev_argsorted_idx = np.argsort(-topic_word_vec)
    for i in range(1, num_topwords_to_check):
        for j in range(i):
            topic_coherence_score += np.log( (eps_to_add+np.sum( 
                                doc_term_matrix[:,topic_rev_argsorted_idx[i]][doc_term_matrix[:,topic_rev_argsorted_idx[j]]] ))  /  \
                                    doc_term_matrix[:,topic_rev_argsorted_idx[j]].sum() )

    # The gensim approach is to normalize by multiplying the coherence by 2/(n*(n-1))        
    return topic_coherence_score, 2.0*topic_coherence_score/(num_topwords_to_check*(num_topwords_to_check-1))



def multitopic_coherence_dense(topic_word_2darray, doc_term_matrix, num_topwords_to_check, eps_to_add=0.000000001, do_sparse_dotprod=True):
    # Accelerated UMass coherence of *multiple* topics computed using a (dense) np.array doc_term_matrix
    # doc_term_matrix = doc_term_matrix.astype(np.bool)

    if not do_sparse_dotprod:
        T2W4coherence = (doc_term_matrix.T.astype(bool).astype(float).dot(doc_term_matrix.astype(bool).astype(float)))  
    else:
        doc_term_matrix_sparse = sparse.csr_matrix(doc_term_matrix)
        T2W4coherence = ( doc_term_matrix_sparse.T.astype(bool).astype(float).dot(doc_term_matrix_sparse.astype(bool).astype(float)))
        T2W4coherence = T2W4coherence.toarray()

    #doc_counts_per_topiclabel is just the diagonal of T2Wbinarysummed so can skip summing
    T2W4coherence = np.divide(T2W4coherence+eps_to_add, T2W4coherence.diagonal()[:,None]) 
    T2W4coherence = np.log(T2W4coherence)
    print('Precomputed coherence matrix')
    alltopics_rev_argsorted_idx = np.array( [np.argsort(-topic_word_vec) for topic_word_vec in topic_word_2darray] )
    all_coherence_scores = np.zeros(len(topic_word_2darray))
    for tidx, topic_rev_argsorted_idx in enumerate(alltopics_rev_argsorted_idx):
        #if ((tidx>0) and (tidx % 100==0)):
            #print('Completed coherence for topic idx', tidx, 'of', len(topic_word_2darray))
        for i in range(1, num_topwords_to_check):
            for j in range(i):
                all_coherence_scores[tidx] += T2W4coherence[topic_rev_argsorted_idx[j], topic_rev_argsorted_idx[i]]

    # The gensim approach is to normalize by multiplying the coherence by 2/(n*(n-1)), so we output that too      
    return all_coherence_scores, 2.0*all_coherence_scores/(num_topwords_to_check*(num_topwords_to_check-1))


