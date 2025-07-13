'''
@author: [ANONYMIZED]
'''
import numpy as np
from datetime import datetime
import os

from src import coherence
from src.preprocessing_util import *

" NOTE: this script must be run from within the OCTIS-Master directory (see our ReadMe.md)"
from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.coherence_metrics import Coherence




" CHOOSE ETM or AVITM HERE "
#############################

# from octis.models.ETM import ETM as TopicModelAlias
# MODEL = 'EmbeddedTM'

from octis.models.ProdLDA import ProdLDA as TopicModelAlias # also known as AVITM
MODEL = 'ProdLDA' # also known as AVITM



# Setup Parameters #
####################
PREPROCESSED_DATA_MAINDIRECTORY = '../preprocessed_data'
OUTPUT_SOLUTION_MAINDIRECTORY = 'solutions'
COHERENCE_OUTPUTS_DIRECTORY = 'coherence_results'
NUM_TOPWORDS_TO_CHECK_COHERENCE_VEC = [5,10,15,20,25]
BASELINE_RUNS = 10
SAVE_RESULTS = True




if __name__ == '__main__':

    for dataname in ['CONGRESS', 'REUTERS', '20NEWSGROUPSPOL']:

        if not SAVE_RESULTS:
            print('\n\nWARNING: not saving results!\n\n')
        RESULTS_CSV_FPATH = "coherenceresults_"+dataname+"_"+MODEL+".csv"

        time_start = datetime.now()
        preprocessed_data_directory = '../preprocessed_data/'+dataname
        D2NGRAM, D2W, D2Windexform, words, ngrams, docs = \
                load_preprocessed_documents(dataname, preprocessed_data_directory, NGRAM_MAX_LENGTH=1)

        # Free memory
        docs_raw_reidx = None 
        D2NGRAM = None 
        D2W = None

        # if MODEL == 'CTM':
        #     # Octis CTM's code saves pickles with generic names. When rerun on different data, 
        #     # it loads without checking which dataset!
        #     # Therefore when running CTM model, always delete its old pickles first!
        #     import os
        #     if os.path.exists("_train.pkl"):
        #       os.remove("_train.pkl")
        #     if os.path.exists("_test.pkl"):
        #       os.remove("_test.pkl")
        #     if os.path.exists("_val.pkl"):
        #       os.remove("val.py")

        # Load a dataset
        dataset = Dataset(corpus = [doc.split(' ') for doc in docs], vocabulary=list(words))

        # Initialize storage
        LDA_coh_output_means = []
        LDA_coh_output_pertopic = []
        # LDA_coh_output_perASSIGNEDtopic = []
        run = 0
        while (run <= (BASELINE_RUNS-1)):

            model = TopicModelAlias(num_topics=100, use_partitions=False)
            if MODEL=='AVITM':
                model.top_words = 25
            model_output = model.train_model(dataset, top_words=25) # Train the model

            # Do all coherences
            print('Computing coherences', datetime.now(), 'dataname', dataname, 'run:', run)

            for coh_topwords_thisister in NUM_TOPWORDS_TO_CHECK_COHERENCE_VEC:
                
                umass = Coherence(texts=dataset.get_corpus(), topk=coh_topwords_thisister, measure='u_mass')
                gensim_coherence_mean =  umass.score(model_output)
                gensim_coherence_per_topic_list = [-1]

                print('\nDataset:', dataname, 'completed run:', run, \
                    'num_topwords_coherence:', coh_topwords_thisister, \
                    'gensim_coherence_mean', gensim_coherence_mean, 'time:', datetime.now())

                LDA_coh_output_means.append( {'num_topwords_to_check':  coh_topwords_thisister, \
                                              'gensim_coherence_umass_mean': gensim_coherence_mean, \
                                              'gensim_coherence_per_topic_list_min': np.min(gensim_coherence_per_topic_list), \
                                              'gensim_coherence_per_topic_list_max': np.max(gensim_coherence_per_topic_list), \
                                              'gensim_coherence_umass_per_topic_list': gensim_coherence_per_topic_list, \
                                              'numtopics_k': len(model_output['topics']), \
                                              'model': MODEL, \
                                              'dataset': dataname, \
                                              'run': run, \
                                              })
                                                  

                # Save results progressively in the loop
                if SAVE_RESULTS:
                    pd.DataFrame(LDA_coh_output_means).to_csv(path_or_buf = RESULTS_CSV_FPATH, index=False)
                    print('\n\ndataset:', dataname, MODEL, ' Coherence COMPLETE in ', \
                         np.round((datetime.now()-time_start).total_seconds()/60.,3), 'mins.', datetime.now())

                else:
                    print("NOT-saving results!")
            run += 1


