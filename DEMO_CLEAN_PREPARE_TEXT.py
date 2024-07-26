'''
@author: Adam Breuer
'''
from src.text_cleaning_util import *
from src.save_load_docs_util import *
from src.ELDA_matrix_building_util import *

from pathlib import Path


PREPROCESSED_DATA_MAINDIRECTORY = 'preprocessed_data'


if __name__ == '__main__':

    ##### LOAD a new text dataset, remove stopwords, etc. 
    ##### We'll demo this with an Australian Broadcasting Co dataset from NLTK.
    ##### (to use your own dataset, create a docs_raw object that is a list where each element is a document) 

    ######  ######  ######  ######  ######  ######   ######   ######
    #             DOWNLOAD THE RAW DATA for PREPROCESSING          #
    ######  ######  ######  ######  ######  ######   ######   ######
    nltk.download('abc') 
    from nltk.corpus import abc
    dataname = 'AUSBROADCASTINGCO'
    try:
        Path("preprocessed_data/"+dataname).mkdir(parents=True, exist_ok=False)
    except:
        print('Preprocessed data directory '+dataname+ 'already exists')
    docs_raw = abc.raw(fileids=[ abc.fileids()[1]]).split('\r\n')
    print('\n\nDownloaded the '+dataname+ ' dataset.')


    ######  ######  ######  ######  ######  ######   ######   ######
    #      PREPROCESS, Remove punct/stopwords, fmt matrix etc.     #
    ######  ######  ######  ######  ######  ######   ######   ######
    PREPROCESSED_DATA_MAINDIRECTORY = 'preprocessed_data'
    start_time_preprocessing = datetime.now()

    docs, docs_raw_reidx = clean_all_documents(docs_raw) # some docs might be empty after cleaning so we drop and reindex docs and corresp. raw (uncleaned) docs

    D2NGRAM, D2W, D2Windexform, words, ngrams = build_docword_matrix(docs, NGRAM_MAX_LENGTH=1)

    print('D2Windexform[3665]', D2Windexform[3665])


    print('Saving cleaned documents and matrix')
    save_preprocessed_documents(dataname, PREPROCESSED_DATA_MAINDIRECTORY+'/'+dataname, D2NGRAM, D2Windexform, D2W, words, ngrams, docs, NGRAM_MAX_LENGTH=1)

    print('\nCOMPLETED DATA PREPROCESSING, STOPWORD REMOVAL, ETC:\n' \
            + "\n"+str(len(docs))+" Preprocessed docs saved in "+PREPROCESSED_DATA_MAINDIRECTORY+'/'+dataname + '\n')
    





