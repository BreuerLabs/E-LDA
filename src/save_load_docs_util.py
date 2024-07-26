'''
@author: Adam Breuer
'''
import numpy as np
from scipy import sparse


def save_preprocessed_documents(dataname, preprocessed_data_directory, D2NGRAM, D2Windexform, \
                                    D2W, words, ngrams, docs, NGRAM_MAX_LENGTH=1):
    with open(preprocessed_data_directory + '/' + dataname + '_D2Windexform.txt', 'w') as outfile:
        for doc in D2Windexform:
            print(*doc, file=outfile)

    with open(preprocessed_data_directory + '/' + dataname + '_words.txt', 'w') as outfile:
        print(*words, file=outfile)

    with open(preprocessed_data_directory + '/' + dataname + '_ngrams_MAXLEN_'+str(NGRAM_MAX_LENGTH)+'.txt', 'w') as outfile:
        print(*ngrams, file=outfile)

    with open(preprocessed_data_directory + '/' + dataname + '_docs.txt', 'w') as outfile:
        for doc in docs:
            print(doc, file=outfile)

    with open(preprocessed_data_directory + '/' + dataname + '_D2Wsparse.txt', 'w') as outfile:
        assert(sparse.issparse(D2W))
        D2W.maxprint = D2W.count_nonzero()
        row_idxs, col_idxs, vals = sparse.find(D2W)
        for ii in range(len(row_idxs)):
            print(str(row_idxs[ii]) + ' ' + str(col_idxs[ii]) + ' ' + str(vals[ii]), file=outfile)

    with open(preprocessed_data_directory + '/' + dataname + '_D2NGRAM_MAXLEN_'+str(NGRAM_MAX_LENGTH)+'.txt', 'w') as outfile:
        assert(sparse.issparse(D2NGRAM))
        D2W.maxprint = D2NGRAM.count_nonzero()
        row_idxs, col_idxs, vals = sparse.find(D2NGRAM)
        for ii in range(len(row_idxs)):
            print(str(row_idxs[ii]) + ' ' + str(col_idxs[ii]) + ' ' + str(vals[ii]), file=outfile)

    print(dataname, ': Saved preprocessed doc-to-word-indices, words, and docs')


def load_preprocessed_documents(dataname, preprocessed_data_directory, NGRAM_MAX_LENGTH=1):
    with open(preprocessed_data_directory + '/' + dataname + '_D2Windexform.txt', 'r') as infile:
        D2Windexform = infile.read().splitlines()
        D2Windexform = np.asarray([[int(x) for x in item.split(' ')] for item in D2Windexform], dtype='object')

    with open(preprocessed_data_directory + '/' + dataname + '_words.txt', 'r') as infile:
        words = np.asarray( infile.read()[:-1].split(' ') ) # there is a \n at the end that must be stripped

    with open(preprocessed_data_directory + '/' + dataname + '_ngrams_MAXLEN_'+str(NGRAM_MAX_LENGTH)+'.txt', 'r') as infile:
        ngrams = np.asarray( infile.read()[:-1].split(' ') ) # there is a \n at the end that must be stripped

    with open(preprocessed_data_directory + '/' + dataname + '_docs.txt', 'r') as infile:
        docs = infile.read().splitlines()

    with open(preprocessed_data_directory + '/' + dataname + '_D2Wsparse.txt', 'r') as infile:
        sparse_inds = infile.read().splitlines()
        sparse_inds = np.asarray([[int(x) for x in row.split(' ')] for row in sparse_inds])
        rows = sparse_inds[:,0].max()+1
        cols = sparse_inds[:,1].max()+1
        D2W = sparse.coo_matrix((sparse_inds[:,2], (sparse_inds[:,0], sparse_inds[:,1])), shape=(rows, cols)).tocsr()   

    with open(preprocessed_data_directory + '/' + dataname + '_D2NGRAM_MAXLEN_'+str(NGRAM_MAX_LENGTH)+'.txt', 'r') as infile:
        sparse_inds = infile.read().splitlines()
        sparse_inds = np.asarray([[int(x) for x in row.split(' ')] for row in sparse_inds])
        rows = sparse_inds[:,0].max()+1
        cols = sparse_inds[:,1].max()+1
        D2NGRAM = sparse.coo_matrix((sparse_inds[:,2], (sparse_inds[:,0], sparse_inds[:,1])), shape=(rows, cols)).tocsr()   
    
    print(dataname, ': Loaded preprocessed doc-to-word-indices, words, and docs')
    return D2NGRAM, D2W, D2Windexform, words, ngrams, docs





