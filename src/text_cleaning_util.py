'''
@author: Adam Breuer
'''
import numpy as np
from scipy import sparse
# import unidecode # useful for accented characters
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
#from nltk.stem import PorterStemmer # or LancasterStemmer, RegexpStemmer, SnowballStemmer
from nltk.stem import SnowballStemmer # or LancasterStemmer, RegexpStemmer, SnowballStemmer
from datetime import datetime


# HERE'S WHERE WE DO ANY CUSTOMIZATION OF PUNCTUAATION
my_punctuation = string.punctuation
#my_punctuation = my_punctuation.replace('-', '')
my_punctuation = my_punctuation.replace('/', '')
my_punctuation += '™'
my_punctuation += "’"


def clean_text(text, my_punctuation, default_stemmer=SnowballStemmer('english'), default_stopwords=stopwords.words('english')):
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
    def remove_singlechar_words(text):
        return  ' '.join( [w for w in text.split() if len(w)>1] )
    # def normalize_accented_characters(text):
    #     tokens = [unidecode.unidecode(accented_string) for accented_string in tokenize_text(text)]
    #     return ' '.join(tokens)
    text = re.sub('/', ' ', text) # words that are slash-separated
    text = text.strip(' ') # strip whitespaces
    text = text.lower() # lowercase
    text = stem_text(text) # stemming
    text = remove_special_characters(text) # remove punctuation and symbols
    text = remove_stopwords(text) # remove stopwords
    text = remove_digitwords(text) # remove stopwords
    text = remove_singlechar_words(text) # remove stopwords
    #text = normalize_accented_characters(text)
    return text





def clean_all_documents(docs_raw):
    docs = [clean_text(this_raw_doc, my_punctuation) for this_raw_doc in docs_raw]
    print('THESE DOCS ARE EMPTY (0-indexed) after cleaning')
    print( [[docs_raw[idx], doc] for idx, doc in enumerate(docs) if doc == ''] )
    print('Cleaned text')
    docs_and_raws = [[docs_raw[idx], doc] for idx, doc in enumerate(docs) if (len(doc)>1)] # remove texts that were empty except for single chars, stopwords, junk
    docs = [docs_and_raw[1] for docs_and_raw in docs_and_raws]
    docs_raw_reidx = [docs_and_raw[0] for docs_and_raw in docs_and_raws] #reindexed because we dropped '' emptys and 1char docs
    return docs, docs_raw_reidx



















