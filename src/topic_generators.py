'''
@author: [ANONYMIZED]
'''

import numpy as np
from scipy import sparse # optional -- set do_sparse_dot_prod=False in functions to avoid (but it usually is much faster w/sparse)



def generate_UMASS_topic2word_matrix_nonlogged(D2W, eps_to_add=0.00001, normalize_to_proper_topics=False, do_sparse_dotprod=True):
    # A UMass topic assigns mass to the element element of φ_{t=space}[v = ‘exploration′] corresponding to the word
    # ‘exploration’ in proportion to the count of documents that contain both ’space’ and ’exploration’,
    # normalized by the count documents that contain ‘space’.
    # D2W should be a Dense NP array. See commented out code below if it's scipy sparse.

    # T2Wbinarysummed = (D2W.T.astype(bool).astype(int).dot(D2W.astype(bool).astype(int))).toarray() + eps_to_add #Original paper did 1.0 but some more recent papers advise to make eps very small
    if not do_sparse_dotprod:
        T2Wbinarysummed = (D2W.T.astype(bool).astype(float).dot(D2W.astype(bool).astype(float)))  #Original paper did 1.0 but some more recent papers advise to make eps very small

    else: 
        D2W_sparse = sparse.csr_matrix(D2W)
        T2Wbinarysummed = ( D2W_sparse.T.astype(bool).astype(float).dot(D2W_sparse.astype(bool).astype(float)))
        T2Wbinarysummed = T2Wbinarysummed.toarray()
        D2W_sparse = None

    #doc_counts_per_topiclabel is just the diagonal of T2Wbinarysummed so can skip summing
    T2W = np.divide(T2Wbinarysummed+eps_to_add, T2Wbinarysummed.diagonal()[:,None]) 
    if normalize_to_proper_topics:
        T2W = T2W/T2W.sum(axis=1)[:,None]
    return( T2W )



# def generate_SIMPLE_topic2word_matrix_nonlogged(D2W, eps_to_add=0.00001, normalize_to_proper_topics=False):
#     numwords = D2W.shape[1]
#     # SIMPLE assigns mass to each word v proportional to the count of times word v appears in documents containing token t,
#     # denoted by Wt(v), normalized by the count of all words except t in documents containing t, denoted by Wt(¬t).

#     # For row t of T2W:
#     #    Take the subset of documents that contain t
#     #    For each word v:
#     #       Add up all of the occurrences of v, which is colsum v of just the rows that contain t
#     #       Divide that by the sum of those colsums, minus the v'th colsum of those rows.

#     numerators = np.dot (D2W.astype(bool).astype(float).T,  D2W.astype(bool).astype(float) )
#     #print('\n\nsimple numerators\n', numerators)
#     doc_lengths = np.sum(D2W, 1)
#     count_of_words_in_docs_that_contain_token = np.dot(doc_lengths, D2W.astype(bool).astype(int))
#     #print('\n\nsimple count_of_words_in_docs_that_contain_token\n', count_of_words_in_docs_that_contain_token)
#     # But this ignores the fact that we are supposed to except the t's from this count
#     # T2W = np.divide(numerators+eps_to_add, count_of_words_in_docs_that_contain_token[:,None]) 
#     denoms = np.repeat(np.array([count_of_words_in_docs_that_contain_token]).reshape((numwords,1)), numwords, axis=1)
#     #print('\ndenoms', denoms)
#     T2W = (numerators+eps_to_add) / denoms
    
#     if normalize_to_proper_topics:
#         T2W = T2W/T2W.sum(axis=1)[:,None]

#     return(T2W)



def generate_SIMPLE_topic2word_matrix_nonlogged(D2W, eps_to_add=0.00001, normalize_to_proper_topics=False, do_sparse_dotprod=True):
    numwords = D2W.shape[1]
    # SIMPLE assigns mass to each word v proportional to the count of times word v appears in documents containing token t,
    # denoted by Wt(v), normalized by the count of all words except t in documents containing t, denoted by Wt(¬t).

    # For row t of T2W:
    #    Take the subset of documents that contain t
    #    For each word v:
    #       Add up all of the occurrences of v, which is colsum v of just the rows that contain t
    #       Divide that by the sum of those colsums, minus the v'th colsum of those rows.
    if not do_sparse_dotprod:
        numerators = np.dot (D2W.astype(bool).astype(float).T,  D2W )
        #print('\n\nsimple numerators\n', numerators)
        doc_lengths = np.sum(D2W, 1)

    else:
        D2W_sparse = sparse.csr_matrix(D2W)
        numerators = (D2W_sparse.T.astype(bool).astype(float).dot(D2W_sparse)).toarray()
        doc_lengths = np.asarray( D2W_sparse.sum(axis=1) )[:,0]
        D2W_sparse = None


    count_of_words_in_docs_that_contain_token = np.dot(doc_lengths, D2W.astype(bool).astype(int))

    #print('\n\nsimple count_of_words_in_docs_that_contain_token\n', count_of_words_in_docs_that_contain_token)
    # But this ignores the fact that we are supposed to except the t's from this count
    # T2W = np.divide(numerators+eps_to_add, count_of_words_in_docs_that_contain_token[:,None]) 
    denoms = np.repeat(np.array([count_of_words_in_docs_that_contain_token]).reshape((numwords,1)), numwords, axis=1)
    #print('\ndenoms', denoms)
    T2W = (numerators+eps_to_add) / denoms
    
    if normalize_to_proper_topics:
        T2W = T2W/T2W.sum(axis=1)[:,None]

    return(T2W)




def generate_DOTPROD_topic2word_matrix_nonlogged(D2W, eps_to_add=0.00001, normalize_to_proper_topics=False, do_sparse_dotprod=True):
    # numwords = D2W.shape[1]
    # T2W = np.dot (D2W.astype(bool).astype(float).T,  D2W.astype(bool).astype(float) ) + eps_to_add

    if not do_sparse_dotprod:
        T2W = np.dot (D2W.astype(bool).astype(float).T,  D2W.astype(bool).astype(float) ) + eps_to_add

    else:
        D2W_sparse = sparse.csr_matrix(D2W)
        T2W = ((D2W_sparse.astype(bool).astype(float).T).dot( D2W_sparse.astype(bool).astype(float) )).toarray() + eps_to_add
        D2W_sparse = None

    if normalize_to_proper_topics:
        T2W = T2W/T2W.sum(axis=1)[:,None]

    return(T2W)





# Old version of simple excludes token t for topic about token t in denominator
# def generate_SIMPLE_topic2word_matrix_nonlogged(D2W, eps_to_add=0.00001, normalize_to_proper_topics=False):
#     numwords = D2W.shape[1]
#     # SIMPLE assigns mass to each word v proportional to the count of times word v appears in documents containing token t,
#     # denoted by Wt(v), normalized by the count of all words except t in documents containing t, denoted by Wt(¬t).

#     # For row t of T2W:
#     #    Take the subset of documents that contain t
#     #    For each word v:
#     #       Add up all of the occurrences of v, which is colsum v of just the rows that contain t
#     #       Divide that by the sum of those colsums, minus the v'th colsum of those rows.

#     numerators = np.dot (D2W.astype(bool).astype(float).T,  D2W )
#     #print('\n\nsimple numerators\n', numerators)
#     doc_lengths = np.sum(D2W, 1)
#     count_of_words_in_docs_that_contain_token = np.dot(doc_lengths, D2W.astype(bool).astype(int))
#     #print('\n\nsimple count_of_words_in_docs_that_contain_token\n', count_of_words_in_docs_that_contain_token)
#     # But this ignores the fact that we are supposed to except the t's from this count
#     # T2W = np.divide(numerators+eps_to_add, count_of_words_in_docs_that_contain_token[:,None]) 
#     denoms = np.repeat(np.array([count_of_words_in_docs_that_contain_token]).reshape((numwords,1)), numwords, axis=1)
#     #print('\ndenoms', denoms)
#     T2W = (numerators+eps_to_add) / (denoms - numerators)
    
#     if normalize_to_proper_topics:
#         T2W = T2W/T2W.sum(axis=1)[:,None]

#     return(T2W)


if __name__ == '__main__':
    print('Unit testing topic generators')
    D2W_unittest = np.asarray([[1,3,0,1],[1,0,0,3],[4,4,5,0]])
    T2W_Umass_unittest = generate_UMASS_topic2word_matrix_nonlogged(D2W_unittest)
    print('\nD2W_unittest\n', D2W_unittest)
    print('\nT2W_Umass_unittest\n', T2W_Umass_unittest)
    assert(T2W_Umass_unittest[3,1] == 5.00005000e-01)

    T2W_SIMPLE_unittest = generate_SIMPLE_topic2word_matrix_nonlogged(D2W_unittest)
    print('\nD2W_unittest\n', D2W_unittest)
    print('\nT2W_SIMPLE_unittest\n', T2W_SIMPLE_unittest)
    print(T2W_SIMPLE_unittest[0,2])
    # assert(T2W_SIMPLE_unittest[0,2] == 0.2941182352941176) # old way removing token t count from denominator
    assert(T2W_SIMPLE_unittest[0,2] == 0.2272731818181818)

    T2W_Umass_unittest = generate_UMASS_topic2word_matrix_nonlogged(D2W_unittest, normalize_to_proper_topics=True)
    print('\nD2W_unittest\n', D2W_unittest)
    print('\nT2W_Umass_unittest\n', T2W_Umass_unittest, np.sum(T2W_Umass_unittest,1))

    T2W_SIMPLE_unittest = generate_SIMPLE_topic2word_matrix_nonlogged(D2W_unittest, normalize_to_proper_topics=True)
    print('\nD2W_unittest\n', D2W_unittest)
    print('\nT2W_SIMPLE_unittest\n', T2W_SIMPLE_unittest, np.sum(T2W_SIMPLE_unittest,1))






