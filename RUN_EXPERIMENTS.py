'''
@author: [Anonymized]
'''
import numpy as np
from datetime import datetime
import gzip
import os
import glob
from time import sleep
from sklearn.feature_extraction.text import CountVectorizer

from src.exemplar_LDA import exemplar_LDA, reconstruct_besttopic_perword_and_objvalues
from src.preprocessing_util import *

MALLET_PATH = '../mallet/bin/mallet' # UPDATE path to your installation of MALLET
from src.Gensim.ldamallet import LdaMallet 



dataname = 'CONGRESS'
# dataname = '20NEWSGROUPSPOL'
# dataname = 'Reuters'
PREPROCESSED_DATA_MAINDIRECTORY = 'preprocessed_data/'
EXPERIMENT_RESULTS_FNAME = "EXPERIMENT_1_Results_" + dataname +".csv"
MALLET_VS_ELDA_STORAGE = [] # Storage for experimental outputs

# Alpha prior on document sparse set of topics
baselines = [1, 0.1, 'auto'] # Gibbs, Gibbs-Sparse, Gibbs-Bayes
numtopics_k = 100
num_runs = 10



if __name__ == '__main__':

    _, D2W, D2Windexform, words, ngrams, docs = \
    load_preprocessed_documents(dataname, PREPROCESSED_DATA_MAINDIRECTORY + dataname, NGRAM_MAX_LENGTH=1)
    _ = None # free memory 

    # Prepare data in gensim format. We do this in the loop to avoid the mallet file location error.
    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(docs) #no we want the clean version of the docs
    (gensim_corpus, gensim_dict) = vect2gensim(vectorizer, sparse.csr_matrix(doc_term_matrix))
    all_corpus_wordidx = [[item[0] for item in doc] for doc in gensim_corpus]
    all_corpus_wordcount = [[item[1] for item in doc] for doc in gensim_corpus]

    # Mallet has its own preprocessing. We need to reindex our data so we can match ours to Mallet's for a fair comparison
    # Gensim vocab dict maps idx (keys) to words (values). We need inverted one to 
    gensim_dict_INVERTED = {v: k for k, v in gensim_dict.items()}


    # Run all baselines + E-LDA for 10 runs each on same (baseline) topics each run + save results
    for baseline_alpha in baselines:
        for run in range(num_runs):
            # Looping many runs of Mallet causes its temp files to build up and cause errors, so we remove these temp files before each run
            print('\n\nClearing old mallet files from user temp folders (Update this with your Mallet temp storage directory)')
            f1 = glob.glob("../../../../../../../var/folders/1n/7134kmg50ts8nbwjm1y_lyb00000gn/T/*_inferencer.mallet*")
            f2 = glob.glob("../../../../../../../var/folders/1n/7134kmg50ts8nbwjm1y_lyb00000gn/T/*_corpus*")
            f3 = glob.glob("../../../../../../../var/folders/1n/7134kmg50ts8nbwjm1y_lyb00000gn/T/*state.mallet*")
            f4 = glob.glob("../../../../../../../var/folders/1n/7134kmg50ts8nbwjm1y_lyb00000gn/T/*_topickeys*")
            f5 = glob.glob("../../../../../../../var/folders/1n/7134kmg50ts8nbwjm1y_lyb00000gn/T/*doctopics*")
            for mallettempfile in f1+f2+f3+f4+f5:
                os.remove(mallettempfile)

            print('Beginning Mallet LDA', datetime.now())
            if baseline_alpha != 'auto':
                lda = LdaMallet(mallet_path=MALLET_PATH, corpus=gensim_corpus, id2word=gensim_dict, \
                    num_topics=100, alpha=int(100*baseline_alpha), optimize_interval=0)
                alphas = [baseline_alpha]
            else:
                lda = LdaMallet(mallet_path=MALLET_PATH, corpus=gensim_corpus, id2word=gensim_dict, \
                    num_topics=100, optimize_interval=1)
            print('Completed Mallet LDA', datetime.now())

            # Get the Gibbs word-topic assignments
            # First the mallet temp file with the various states of the model
            fpath_to_gzipped_mallet_topicword_assignments = lda.fstate()
            sleep(8) # Wait for Mallet to save its assignments
            malletassigns = pd.read_csv(fpath_to_gzipped_mallet_topicword_assignments, sep=' ', header=None,skiprows = [0,1,2]) #top rows = header,alphas,beta
            malletassigns.columns = ['doc', 'source', 'pos', 'typeindex', 'type', 'topic']
            doc_lengths_mallet = malletassigns.groupby( [ "doc"] ).size().to_frame(name = 'doclength_mallet').reset_index()

            # Ensure all assignments are non-null and get compute Mallet objective value
            word_idx_gensimdict_newway = [gensim_dict_INVERTED[x] if not pd.isnull(x) else np.nan for x in malletassigns['type'].values]
            malletassigns['word_idx_gensimdict_newway'] = np.asarray(word_idx_gensimdict_newway).astype(int)
            topics = np.asarray( lda.get_topics() )
            topics_idx = malletassigns['topic'].values
            words_idx = malletassigns['word_idx_gensimdict_newway'].values
            idx_pairs = [[topics_idx[idx], words_idx[idx]] for idx in range(len(topics_idx))]
            phis = np.asarray([topics[pair[0]][pair[1]] for pair in idx_pairs])
            malletassigns['phi'] = phis
            sum_of_logphis_mallet = np.sum(np.log(phis[phis>0]))
            print('\n\n\nMALLET Objective Value on '+dataname+' for baseline '+str(baseline_alpha)+' on run '+str(run)+':', sum_of_logphis_mallet)

             # Reindex our data so that the word ordering matches Mallet topic word order and drop words that Mallet prunes
            set_of_words_missing_from_mallet = set(words).difference(set(malletassigns['type'].values))
            mallet_missing_wordIDSpergensim_set = set(list(range(len(words)))).difference(set(word_idx_gensimdict_newway))
            missing_words_toipcprob_sums = [np.sum(topics[:,idx]) for idx in mallet_missing_wordIDSpergensim_set]
            unique_topic_counts_perdoc = malletassigns[['doc', 'topic']].groupby(['doc']).agg(['nunique'])
            mean_unq_topics_assigned_perdoc = unique_topic_counts_perdoc.mean()
            sumoverdocs_num_unique_topics_in_wordtopicassignments_perdoc = unique_topic_counts_perdoc.sum()
            unique_topics_assigned = np.unique(malletassigns['topic'].values)
            words_gensim  = [gensim_dict[idx] for idx in range(len(gensim_dict)) ]
            assert(np.all(np.sort(words) == np.sort(words_gensim))) # Test that we reindexed for a perfect match
            temp = sorted(words_gensim, key=list(words).index) # Sorted List. 
            reindex = list(map(lambda x: words_gensim.index(x), temp)) # Get the index to sort the list
            topics_sorted_to_my_words = topics[:, reindex] # Re sort each Mallet topic so to match our Vocab word-ordering

            # Remove the words that Mallet pruned due to its preprocessing from our vocabulary so we match
            miss_widxs_all = []
            for missword in set_of_words_missing_from_mallet:
                miss_widxs_all.append(  np.where(missword == words)[0][0]  )
            D2Windexform_exclmisswords = []
            for doc in D2Windexform:
                D2Windexform_exclmisswords.append([w for w in doc if w not in miss_widxs_all])




            ############     ##############   We are now ready to run E-LDA   ############     ############## 
            time_start = datetime.now()
            topics_sorted_to_my_words[topics_sorted_to_my_words==0] = 10**-10
            T2W = np.log(topics_sorted_to_my_words)
            numdocs = len(D2Windexform)
            T2W[T2W == -np.inf] = 0 # Take care of mallet's pruned words in the topic distribution matrix
            T2W = T2W.astype(np.float64)
            kappaD = int( numdocs * mean_unq_topics_assigned_perdoc ) # A bit funny because mallet dropped some docs

            solution_set_S, ELDA_objective_value = exemplar_LDA(D2Windexform_exclmisswords, T2W, kappaD, time_start)
            print('\nE-LDA Objective Value:', ELDA_objective_value)




            ############     ##############   RandK Random Baseline   ############     ############## 
            # Now we compute value of a random solution WHERE: the Gibbs topics were randomly reindexed. 
            random_objvals_mallet = []
            for randomdraw in range(30): # 30 trials of random reassignment to get a good benchmark
                malletassigns['randomtopic'] = malletassigns['topic'] + np.random.choice(list(range(1,(numtopics_k+1)))) # now topic indices are random
                malletassigns.loc[malletassigns['randomtopic']>=numtopics_k, 'randomtopic'] = malletassigns['randomtopic']-numtopics_k # wrap around as max index is 100
                topics_idx = malletassigns['randomtopic'].values
                words_idx = malletassigns['word_idx_gensimdict_newway'].values
                idx_pairs = [[topics_idx[idx], words_idx[idx]] for idx in range(len(topics_idx))]
                randomphis = np.asarray([topics[pair[0]][pair[1]] for pair in idx_pairs])
                randomphis[randomphis<=0] = 10**-10
                sum_of_randomlogphis_mallet = np.sum(np.log(randomphis))
                random_objvals_mallet.append(sum_of_randomlogphis_mallet)
            AVG_sum_of_randomlogphis_mallet = np.mean(random_objvals_mallet)
            print('**RandK** Mean Objective Value (30 redraws) on '+dataname+' on run '+str(run)+':', AVG_sum_of_randomlogphis_mallet)


            # Store each run's results
            MALLET_VS_ELDA_STORAGE.append( {'numtopics_k':  numtopics_k, \
                                          'alpha': baseline_alpha, \
                                          'alphas': alphas, \
                                          'run': run, \
                                          'dataset': dataname, \
                                          'ELDA_Objective': ELDA_objective_value, \
                                          'MALLET_Objective': sum_of_logphis_mallet, \
                                          'MALLET_RANDOMK_Objective_avg': AVG_sum_of_randomlogphis_mallet,
                                          'MALLET_RANDOMK_Objective_alldraws': random_objvals_mallet,
                                          'mean_unq_topics_assigned_perdoc': mean_unq_topics_assigned_perdoc, \
                                        } )

            # Progressively save results during the loop
            pd.DataFrame(MALLET_VS_ELDA_STORAGE).to_csv(path_or_buf = EXPERIMENT_RESULTS_FNAME, index=False)










