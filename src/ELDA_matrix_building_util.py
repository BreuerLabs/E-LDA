'''
@author: Adam Breuer
'''
import pandas as pd
import numpy as np
# import unidecode
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer



def vect2gensim(vectorizer, dtmatrix):
    import gensim # inside func def as only needed if you want to run gensim and mallet models to compare
    from gensim.corpora.dictionary import Dictionary
     # transform sparse matrix into gensim corpus and dictionary
    corpus_vect_gensim = gensim.matutils.Sparse2Corpus(dtmatrix, documents_columns=False)
    dictionary = Dictionary.from_corpus(corpus_vect_gensim,
        id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))

    return (corpus_vect_gensim, dictionary)


def build_docword_matrix(docs, NGRAM_MAX_LENGTH=1):
    'Accepts CLEANED docs i.e. list of strings where each string is a cleaned doc'

    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, NGRAM_MAX_LENGTH))
    D2NGRAM = vectorizer.fit_transform(docs)
    print('Computed doc-word matrix')

    ngrams = vectorizer.get_feature_names_out() # includes ngrams
    ngrams = np.asarray([ngram.replace(' ', '_') for ngram in ngrams]) # replace spaces in ngrams with underscore _
    # Sort words by corpus frequency so common words (actually common ngrams) are proximal in memory for speed
    ngram_freq_argsort = list(np.argsort(np.asarray(D2NGRAM.sum(axis=0))[0]))[::-1]
    D2NGRAM = D2NGRAM.toarray()[:,ngram_freq_argsort]
    ngrams = ngrams[ngram_freq_argsort]
    words_inds = [idx for idx, v in enumerate(ngrams) if '_' not in v] # remove 2-grams, 3-grams, etc.
    words = np.asarray( ngrams[words_inds] ) # drop >1-grams from wordlist
    D2W = D2NGRAM[:,words_inds]  # drop >1-grams from doc 2 word matrix

    print('Tokenization preprocessing complete')
    numdocs, numwords = D2W.shape
    D2Windexform = []
    for doc_id in range(numdocs):
        D2Windexform.append([])
        for word_id in range(numwords):
            for ii in range(D2W[doc_id, word_id]):
                D2Windexform[doc_id].append(word_id)

    # re-sparsify D2W for saving
    D2W = sparse.csr_matrix(D2W)
    D2NGRAM = sparse.csr_matrix(D2NGRAM)

    return D2NGRAM, D2W, D2Windexform, words, ngrams 


def save_preprocessed_topics(dataname, GENERATE_TOPICCANDIDATES_USING, \
                                preprocessed_data_directory, T2W, NGRAM_MAX_LENGTH=1):
    with open(preprocessed_data_directory+'/'+dataname+'_topics_'+GENERATE_TOPICCANDIDATES_USING+\
                            '_NGRAMMAXLEN'+str(NGRAM_MAX_LENGTH)+'.txt', 'w') as outfile:
        for topic in T2W:
            print(*topic, file=outfile)
    print(dataname, ': Saved preprocessed topic-word matrix for topics generated via', GENERATE_TOPICCANDIDATES_USING)


def save_preprocessed_coherencepertopic(dataname, GENERATE_TOPICCANDIDATES_USING, \
                                COHERENCE_OUTPUTS_DIRECTORY, coherences_all_topicid_x_numtopwords,\
                                NUM_TOPWORDS_TO_CHECK_COHERENCE_VEC, NGRAM_MAX_LENGTH=1):
    colnames = ['word'] + [str(x) for x in NUM_TOPWORDS_TO_CHECK_COHERENCE_VEC]
    coherence_df = pd.DataFrame(coherences_all_topicid_x_numtopwords, columns = colnames)
    fname = COHERENCE_OUTPUTS_DIRECTORY+'/'+dataname+'_CoherencesAllTopics_'+GENERATE_TOPICCANDIDATES_USING+\
                            '_NGRAMMAXLEN'+str(NGRAM_MAX_LENGTH)+'.csv'
    coherence_df.to_csv(fname, index=False)
    print(dataname, ': Saved coherences for all topics')


def load_preprocessed_topics(dataname, GENERATE_TOPICCANDIDATES_USING, \
                                preprocessed_data_directory, NGRAM_MAX_LENGTH=1):
    with open(preprocessed_data_directory+'/'+dataname+'_topics_'+GENERATE_TOPICCANDIDATES_USING+\
                                            '_NGRAMMAXLEN'+str(NGRAM_MAX_LENGTH)+'.txt', 'r') as infile:
        T2W = infile.read().splitlines()
        T2W = np.asarray([[np.float(x) for x in item.split(' ')] for item in T2W])

    print(dataname, ': Loaded preprocessed topic-to-word matrix')
    return T2W


def load_solution_set_S(filepath):
    with open(filepath) as infile:
        solution_set_S = infile.read().splitlines()
        solution_set_S = np.asarray([[int(x) for x in item.split(' ')] for item in solution_set_S])
    print(filepath, ': Loaded solution_set_S')
    return solution_set_S























