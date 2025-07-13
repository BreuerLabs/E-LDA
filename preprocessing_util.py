'''
@author: [ANONYMIZED]
'''


import pandas as pd
import numpy as np
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer # or LancasterStemmer, RegexpStemmer, SnowballStemmer
# import unidecode # useful for accents
from datetime import datetime
from scipy import sparse

############# GENSIM
import gensim
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import CountVectorizer


# https://stackoverflow.com/questions/48865150/pipeline-for-text-cleaning-processing-in-python
#default_stemmer = PorterStemmer()
default_stemmer = SnowballStemmer('english')
default_stopwords = stopwords.words('english') # or any other list of your choice

# HERE's WHERE WE DO ANY CUSTOMIZATION OF PUNCTUAATION
my_punctuation = string.punctuation
#my_punctuation = my_punctuation.replace('-', '')
my_punctuation = my_punctuation.replace('/', '')
my_punctuation += '™'
my_punctuation += "’"


def vect2gensim(vectorizer, dtmatrix):
     # transform sparse matrix into gensim corpus and dictionary
    corpus_vect_gensim = gensim.matutils.Sparse2Corpus(dtmatrix, documents_columns=False)
    dictionary = Dictionary.from_corpus(corpus_vect_gensim,
        id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))

    return (corpus_vect_gensim, dictionary)


def clean_text(text, my_punctuation):
    def tokenize_text(text):
        return [w for s in sent_tokenize(text) for w in word_tokenize(s)]
    # NOTE this leaves in hyphen and /. The hyphen is for hyphenated words. The '/' is for web addresses and stuff.
    def remove_special_characters(text, characters=my_punctuation):
        tokens = tokenize_text(text)
        pattern = re.compile('[{}]'.format(re.escape(characters)))
        return ' '.join(filter(None, [pattern.sub('', t) for t in tokens]))
    def stem_text(text, stemmer=default_stemmer):
        tokens = tokenize_text(text)
        return ' '.join([stemmer.stem(t) for t in tokens])
    def remove_stopwords(text, stop_words=default_stopwords):
        tokens = [w for w in tokenize_text(text) if w not in stop_words]
        return ' '.join(tokens)
    def remove_digitwords(text):
        tokens = [w for w in tokenize_text(text) if not any(i.isdigit() for i in w)]
        return ' '.join(tokens)
    # def normalize_accented_characters(text):
    #     tokens = [unidecode.unidecode(accented_string) for accented_string in tokenize_text(text)]
    #     return ' '.join(tokens)

    text = text.strip(' ') # strip whitespaces
    text = text.lower() # lowercase
    text = re.sub('/', ' ', text) # words that are slash-separated THIS I NEW
    text = stem_text(text) # stemming
    text = remove_special_characters(text) # remove punctuation and symbols
    text = remove_stopwords(text) # remove stopwords
    text = remove_digitwords(text) # remove stopwords
    #text = normalize_accented_characters(text)
    #text.strip(' ') # strip whitespaces again?
    return text


def preprocess_documents(docs_raw, NGRAM_MAX_LENGTH=1):
    docs = [clean_text(this_raw_doc, my_punctuation) for this_raw_doc in docs_raw]
    print('THESE DOCS ARE EMPTY (0-indexed) after cleaning')


    print( [[docs_raw[idx], doc] for idx, doc in enumerate(docs) if doc == ''] )


    print('Cleaned text')
    docs_and_raws = [[docs_raw[idx], doc] for idx, doc in enumerate(docs) if doc != ''] # remove texts that were empty except for characters and stopwords and junk
    docs = [docs_and_raw[1] for docs_and_raw in docs_and_raws]
    docs_raw_reidx = [docs_and_raw[0] for docs_and_raw in docs_and_raws] #reindexed because we dropped '' emptys
    # print('PUNCTUATION PREPROCESSING TIME: ', (datetime.now() - start_time_preprocessing).total_seconds()/60., 'minutes for ', len(docs), 'documents in dataset', dataname)

    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, NGRAM_MAX_LENGTH))
    D2NGRAM = vectorizer.fit_transform(docs)
    print('Computed doc-word matrix')

    ngrams = vectorizer.get_feature_names() # includes ngrams
    ngrams = np.asarray([ngram.replace(' ', '_') for ngram in ngrams]) # replace spaces in ngrams with underscore _
    
    # Sort words by corpus frequency so common words (actually common ngrams) are proximal in memory for speed
    ngram_freq_argsort = list(np.argsort(np.asarray(D2NGRAM.sum(axis=0))[0]))[::-1]
    D2NGRAM = D2NGRAM.toarray()[:,ngram_freq_argsort]
    ngrams = ngrams[ngram_freq_argsort]

    rowsums = D2NGRAM.sum(axis=1)
    # print('THESE DOCS ARE ALSO EMPTY (0-indexed) after cleaning')
    # print( np.where(D2NGRAM[rowsums==0])[0] )
    # ghghghghgh

    D2NGRAM = D2NGRAM[rowsums>0] # remove texts that were empty except for single characters
    docs = [d for idx, d in enumerate(docs) if rowsums[idx]>0] # remove texts that were empty except for single characters
    docs_raw_reidx = [d for idx, d in enumerate(docs_raw_reidx) if rowsums[idx]>0] # remove texts that were empty except for single characters

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

    return D2NGRAM, D2W, D2Windexform, words, ngrams, docs, docs_raw_reidx 
#              <span class="post--timestamp">4 days ago</span>
#               <span class="impressions--count">107284</span>



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
        #sparse.save_npz(preprocessed_data_directory + '/' + dataname + '_D2Wsparse.npz', D2W)
        # your_matrix_back = sparse.load_npz("yourmatrix.npz")
    print(dataname, ': Saved preprocessed doc-to-word-indices, words, and docs')



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



def load_preprocessed_documents(dataname, preprocessed_data_directory, NGRAM_MAX_LENGTH=1):
    with open(preprocessed_data_directory + '/' + dataname + '_D2Windexform.txt', 'r') as infile:
        # print(preprocessed_data_directory + '/' + dataname + '_D2Windexform.txt')
        D2Windexform = infile.read().splitlines()
        # print('\n\n')
        # print(D2Windexform)
        # print('\n\n')

        # for idx, row in enumerate(D2Windexform):
        #     print('\n')
        #     print(row)
        #     for x in row.split(' '):
        #         print(dataname)
        #         print('row', row, 'row_idx', idx)
        #         print(x, 'is x')
        #         print(int(x))
        #     #print([int(x) for x in row.split(' ')])
        # for lineidx, item in enumerate(D2Windexform):
        #     print('line', lineidx, 'item', item)
        #     for x in item.split(' '):
        #         print('x', x)
        #         int(x)
        # np.asarray([[int(x) for x in item.split(' ')] for item in D2Windexform])
        # D2Windexform = np.asarray([[int(x) for x in item.split(' ')] for item in D2Windexform])
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



def clean_parlerpost(raw_post):
    try:
        post = raw_post.split('<div class="card--body">')[1].split('<div')[0]
        post = re.sub('\n', '', post)
        # post = re.sub('</p>','',out)
        # post = re.sub('<p>','',out)
        # post = re.sub('</span>', '',out)
        # post = re.sub('<span>', '',out)
        # post = re.sub('<div>', '',out)
        # post = re.sub('</div>', '',out)
        # out = re.sub('<br>', '', out)
        # post = re.sub('</br>', '', out)
        post = re.sub('<[^>]+>', '', post)

        date = -1
        impressions = -1
        # also try to get date and impressions
        try:
            date = raw_post.split('<span class="post--timestamp">')[1].split('</span>')[0]
        except:
            pass
        try:
            impressions = raw_post.split('<span class="impressions--count">')[1].split('</span>')[0]
        except:
            pass
        return([post, date, impressions])
    except:
        return(False)


def simplify_parlerposts(list_of_clean_parlerpost_outs):

    texts = [l[0] for l in list_of_clean_parlerpost_outs]

    trumps = [any(x in t.lower() for x in ['trump', 'president', 'support']) for t in texts]

    frauds = [any(x in t.lower() for x in ['georgia', 'fraud', 'count']) for t in texts]

    antifas = [any(x in t.lower() for x in ['blm', 'antifa']) for t in texts]

    dates = [l[1] for l in list_of_clean_parlerpost_outs]
    dates = [re.sub(' days ago', '', d) for d in dates]
    dates = [re.sub('1 month ago', '30', d) for d in dates]
    dates = [re.sub('2 months ago', '60', d) for d in dates]
    dates = [re.sub('3 months ago', '90', d) for d in dates]
    dates = [re.sub('4 months ago', '90', d) for d in dates]

    dates = [re.sub('1 week ago', '7', d) for d in dates]
    dates = [re.sub('2 weeks ago', '14', d) for d in dates]
    dates = [re.sub('3 weeks ago', '21', d) for d in dates]
    dates = [re.sub('4 weeks ago', '21', d) for d in dates]
    dates_parsed = []
    for d in dates:
        try:
            (dates_parsed.append(int(d)))
        except:
            dates_parsed.append(-1)




    impressions = [int(l[2]) for l in list_of_clean_parlerpost_outs]

    d = pd.DataFrame({  'trumps':  trumps, \
                        'frauds': frauds, \
                        'antifas': antifas, \
                        'dates': dates_parsed, \
                        'impressions': impressions \
                                    })
    return(d)























