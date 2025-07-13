'''
@author: [ANONYMIZED] 
'''
import numpy as np
from datetime import datetime

from src import coherence
from src.preprocessing_util import *

from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse

from bertopic import BERTopic
import gensim



PREPROCESSED_DATA_MAINDIRECTORY = 'preprocessed_data'
NUM_TOPWORDS_TO_CHECK_COHERENCE_VEC = [5,10,15,20,25]
SAVE_RESULTS = True
LDA_RUNS = 10
MODEL = 'BERTopic'


if __name__ == '__main__':

    for dataname in ['CONGRESS', 'REUTERS', '20NEWSGROUPSPOL']:

        if not SAVE_RESULTS:
            print('\n\nWARNING: not saving results!\n\n')
        RESULTS_CSV_FPATH = "coherenceresults_"+dataname+"_"+MODEL+".csv"


        time_start = datetime.now()
        preprocessed_data_directory = 'preprocessed_data/'+dataname
        D2NGRAM, D2W, D2Windexform, words, ngrams, docs = \
                load_preprocessed_documents(dataname, preprocessed_data_directory, NGRAM_MAX_LENGTH=1)
        numdocs = len(D2Windexform)
        numwords = len(words)
        docs_raw_reidx = None # Free memory
        D2NGRAM = None # Free memory
        D2W = None

        # Initialize storage
        LDA_coh_output_means = []
        LDA_coh_output_pertopic = []
        run = 0
        while (run <= (LDA_RUNS-1)):

            # Prepare data in gensim format. We do this in the loop to avoid the mallet file location error.
            vectorizer = CountVectorizer()
            doc_term_matrix = vectorizer.fit_transform(docs) #no we want the clean version of the docs
            (gensim_corpus, gensim_dict) = vect2gensim(vectorizer, sparse.csr_matrix(doc_term_matrix))

            # Do all coherences
            print('Computing coherences', datetime.now(), 'dataname', dataname, 'run:', run)

            for coh_topwords_thisister in NUM_TOPWORDS_TO_CHECK_COHERENCE_VEC:

                """ RUN BERTOPIC HERE """
                BERTopic_model = BERTopic(top_n_words=coh_topwords_thisister)
                print('starting fit')
                topics, probs = BERTopic_model.fit_transform(docs)
                BERTopics = BERTopic_model.get_topics()
                BERTopics_wordsorted = []

                for tid, keyvals in BERTopics.items():
                    BERTopics_wordsorted.append ([keyval[0] if len(keyval[0]) else 'placeholderzeroprobword' for keyval in keyvals])  #some '' 

                gensim_coherence_object = gensim.models.coherencemodel.CoherenceModel(topics=BERTopics_wordsorted, \
                        dictionary=gensim_dict, corpus=gensim_corpus, coherence='u_mass', topn=coh_topwords_thisister)
                gensim_coherence_per_topic_list = gensim_coherence_object.get_coherence_per_topic()                    
                gensim_coherence_mean = gensim_coherence_object.get_coherence()

                print('\nDataset:', dataname, 'completed run:', run, 'num_topwords_coherence:', \
                        coh_topwords_thisister, 'gensim_coherence_mean', gensim_coherence_mean, 'time:', datetime.now())

                LDA_coh_output_means.append( {'num_topwords_to_check':  coh_topwords_thisister, \
                                              'gensim_coherence_umass_mean': gensim_coherence_mean, \
                                              'gensim_coherence_per_topic_list_min': np.min(gensim_coherence_per_topic_list), \
                                              'gensim_coherence_per_topic_list_max': np.max(gensim_coherence_per_topic_list), \
                                              'gensim_coherence_umass_per_topic_list': gensim_coherence_per_topic_list, \
                                              'numtopics_k': len(BERTopics_wordsorted), \
                                              'model': MODEL, \
                                              'dataset': dataname, \
                                              'run': run, \
                                            } )

                # Save results progressively in the loop
                if SAVE_RESULTS:
                    pd.DataFrame(LDA_coh_output_means).to_csv(path_or_buf = RESULTS_CSV_FPATH, index=False)
                    print('\n\ndataset:', dataname,' BERTopic Coherence COMPLETE in ', \
                        np.round((datetime.now()-time_start).total_seconds()/60.,3), 'mins.', datetime.now())
                else:
                    print("NOT-saving results!")

            run += 1


