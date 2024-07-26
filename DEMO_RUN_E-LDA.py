'''
@author: [ANONYMIZED]
'''
import numpy as np
import pandas as pd # only necessary to save csv of analyzed solution objective values
from datetime import datetime

from src.save_load_docs_util import *
from src.topic_generators import *
from src.exemplar_LDA import exemplar_LDA, reconstruct_besttopic_perword_and_objvalues

from src import coherence # OPTIONAL


PREPROCESSED_DATA_MAINDIRECTORY = 'preprocessed_data'
SAVE_TOPICS = False # Slow! it's a large dense matrix
DO_MANUAL_COHERENCE = True
NUM_TOPWORDS_TO_CHECK_COHERENCE_VEC = [2,5,10,15,20,25] # To replicate data for plots in Experiments Set 2
DO_JUST_100_TEXTS_FOR_TESTING = False

dataname = 'AUSBROADCASTINGCO'
# dataname = 'CONGRESS'
# dataname = '20NEWSGROUPSPOL'
# dataname = 'REUTERS'

# TOPIC_GENERATOR = 'LOG_DOTPROD' # Called RAW-UMASS in the paper
# TOPIC_GENERATOR = 'RAW_DOTPROD' # Called EXP-UMASS in the paper
TOPIC_GENERATOR = 'FREQSCALE_DOTPROD' # Called CO-OCCURRENCE in the paper

avg_k_per_doc = 3 # kappa parameter in the paper i.e. mean number of topics assigned per doc



if __name__ == '__main__':

    # Load the cleaned data (the important one is the D2Windexform doc-word matrix. We use it in index form as indexing is faster than dot product)
    D2NGRAM, D2W, D2Windexform, words, ngrams, docs, = \
        load_preprocessed_documents(dataname, PREPROCESSED_DATA_MAINDIRECTORY+'/'+dataname, NGRAM_MAX_LENGTH=1)

    # Free memory
    D2NGRAM = None # Free Memory
    docs_raw_reidx = None  # Free memory
    D2NGRAM = None  # Free memory
    if DO_JUST_100_TEXTS_FOR_TESTING:
        D2Windexform = D2Windexform[:100]
        D2W = D2W[:100]
    numdocs = len(docs)

    k = int( numdocs * avg_k_per_doc )  # kappa i.e. upperbound


    ######  ######  ######  ######  ######  ######
    #        Precompute Candidate Topics       ###
    ######  ######  ######  ######  ######  ######
    time_start = datetime.now()

    # Called RAW-UMASS in the paper
    time_start_topicprecompute = datetime.now()
    if TOPIC_GENERATOR == 'LOG_DOTPROD':
        T2W = generate_DOTPROD_topic2word_matrix_nonlogged(D2W, normalize_to_proper_topics=True)
        print('\nCompleted generation of normalized DOTPROD candidate topics')

    # Called EXP-UMASS in the paper
    elif TOPIC_GENERATOR == 'RAW_DOTPROD': 
        T2W = generate_DOTPROD_topic2word_matrix_nonlogged(D2W, normalize_to_proper_topics=False)
        # T2W = np.exp(T2W.astype(np.float128)) # float128 solves overflow for exp(1234) and larger...
        T2W = T2W/T2W.sum(axis=1)[:,None]
        print('\nCompleted generation of normalized RAW_DOTPROD candidate topics')

        # # Less mem way (but slower) by converting row-by-row. 29GB ram instead of 100+. 
        # # Note np.float128 OK because we are counting (smaller) documents that contain word, not instances of word in docs.
        # T2W = generate_DOTPROD_topic2word_matrix_nonlogged(D2W, normalize_to_proper_topics=False)
        # print(T2W.max())
        # T2W = [row for row in T2W]
        # for idx, row in enumerate(T2W):
        #     x = np.exp(row.astype(np.float128))
        #     T2W[idx] = (x / x.sum()).astype(np.float64) # could also do everything here: np.log( (this + 0.0000001) / 1.0000001 )
        #     if ((idx % 100) == 0):
        #         print('generated topic', idx, 'of', len(T2W))
        # T2W = np.asarray(T2W)
        # T2W = T2W + 0.0000000000000001 # so we don't take log(0)
        # T2W = T2W/T2W.sum(axis=1)[:,None]  # Re-normalize so topics are proper (do this 2nd time here so don't lose floating point prec above)

    # Called CO-OCCURRENCE in the paper
    elif TOPIC_GENERATOR == 'FREQSCALE_DOTPROD': #7
        T2W = generate_DOTPROD_topic2word_matrix_nonlogged(D2W, normalize_to_proper_topics=False)
        print("""\n\n
                NOTE! For this topic generator, the topic-word matrix is T2W_topics, 
                but we optimize using the T2W matrix (as usual), 
                where each topic is scaled as the doc-topic distributions 
                are still a point on the simplex. but NOT a centered point. 
                we skip the (computationally tedious) normalization here, 
                as it does not affect the solution topic-word assignments or the coherence.""")

    # Log topic-word matrix elements if they aren't logged
    if TOPIC_GENERATOR not in ['FREQSCALE_DOTPROD', 'RAW_SIMPLENUMONLY']:  
        print('\n\n!!WARNING!! double check that we want to take this log and haven\'t taken it in the topic generator already!!\n\n')
        T2W = np.log(T2W)
    T2W = T2W.astype(np.float64) #in case we used to float128 for exp, this saves memory no prob as topics now normalized==smaller

    ## Collapse identical topics
    # i.e. remove identical topics and update the keywords so if "color"" and "colour" topics are identical
    # then we remove the "colour" topic from T2W and update the "color" topic keyword to "color---colour")
    numtopics_precollapse = len(T2W)
    T2W, unique_topic_idx, identical_topic_idx = np.unique(T2W, axis=0, return_index=True, return_inverse=True)
    keywords_collapsed = [ [] for _ in range(len(unique_topic_idx))]
    for w_idx in range(numtopics_precollapse):
        keywords_collapsed[identical_topic_idx[w_idx]].append(words[w_idx])

    keywords_collapsed = np.asarray(['---'.join(keywords) for keywords in keywords_collapsed])
    print("""\n\nCollapsed identical topics and concatenated their topic keywords. 
            """+str(numtopics_precollapse)+ ' topics collapsed to '+str(len(keywords_collapsed)))

    if SAVE_TOPICS:
        save_preprocessed_topics(dataname, TOPIC_GENERATOR, \
                                    PREPROCESSED_DATA_MAINDIRECTORY+'/'+dataname, T2W, NGRAM_MAX_LENGTH=1)
        print('Saved logged candidate topics (saving/loading is slow as topic-word matrix is large and dense!)')
    print('Elapsed:', np.round((datetime.now()-time_start_topicprecompute).total_seconds()/60.,4), \
            'minutes loading data & generating topic-word mat')



    ######  ######  ######  ######  ######  ######
    #                Run E-LDA                    #
    ######  ######  ######  ######  ######  ######
    solution_set_S, objective_value = exemplar_LDA(D2Windexform, T2W, k, time_start)
    print('\n\nE-LDA OPTIMIZATION Complete. Obtained objective_value:', objective_value,'\n\n')
    solution_set_S = solution_set_S.astype(int)



    ######  ######  ######  ######  ######  ######
    # Analyze and save E-LDA solution (optional) #
    ######  ######  ######  ######  ######  ######

    # Analyze objective values
    print("\n\nAnalyzing solution. This will be slow if topic uniqueness analysis is enabled.")
    objective_values_perround, curr_besttopic_perword_indoc_DxWidthIndexy, curr_bestvals_perword_indoc_DxWidthIndexy = \
        reconstruct_besttopic_perword_and_objvalues(solution_set_S, D2Windexform, T2W)
    print(objective_values_perround[-1])

    # Format and save solution
    objective_val_df = pd.DataFrame({  'iter': np.maximum(np.zeros(len(solution_set_S)), \
                                            (np.array(range(len(solution_set_S))) - numdocs + 1)).astype(int), \
                                        'doc_id':  solution_set_S[:,0], \
                                        'topic_id': solution_set_S[:,1], \
                                        'topic_keyword': keywords_collapsed[solution_set_S[:,1]], \
                                        'objective_val': objective_values_perround, \
                                        'mean_num_topics_perdoc': (np.arange(1,len(solution_set_S)+1))/numdocs, \
                                        })

    objective_val_df.to_csv('RESULT_E-LDA_solution_'+dataname+'__'+TOPIC_GENERATOR+'_ANALYZED.csv', index=False)
    print('\n\n', objective_val_df)
    print('\nFormatted and saved solution to file: ' + 'E-LDA_solution_'+dataname+'__'+TOPIC_GENERATOR+'_ANALYZED.csv')



    ######   ######   ######   ######   ######   ######
    # Topic Keyword analysis per document (optional)  #
    ######   ######   ######   ######   ######   ######
    # Retrieve the topic keywords and associated values for each document
    curr_besttopicKEYWORD_perword_indoc_DxWidthIndexy = [np.asarray(keywords_collapsed)[d] for d in curr_besttopic_perword_indoc_DxWidthIndexy]
    curr_besttopicKEYIDX_UNQ_perdoc = [np.unique(doc_topics) for doc_topics in curr_besttopic_perword_indoc_DxWidthIndexy]
    curr_besttopicKEYWORD_UNQ_perdoc = [np.unique(doc_topickeywords) for doc_topickeywords in curr_besttopicKEYWORD_perword_indoc_DxWidthIndexy]
    curr_besttopicKEYWORD_UNQ_VALUES_perdoc = [np.bincount(curr_besttopic_perword_indoc_DxWidthIndexy[d_idx], \
                                                weights=curr_bestvals_perword_indoc_DxWidthIndexy[d_idx])[curr_besttopicKEYIDX_UNQ_perdoc[d_idx]] \
                                                    for d_idx in range(numdocs)]

    # Sort them by how much each topic contributes to the document and save in a convenient CSV format
    val_sort_indices = [np.argsort(doc_topicvals)[::-1] for doc_topicvals in curr_besttopicKEYWORD_UNQ_VALUES_perdoc]
    curr_besttopicKEYWORD_UNQ_perdoc = [doc_topickeywords[val_sort_indices[idx]] for idx, doc_topickeywords in enumerate(curr_besttopicKEYWORD_UNQ_perdoc)]
    curr_besttopicKEYWORD_UNQ_VALUES_perdoc = [doc_topicvals[val_sort_indices[idx]] for idx, doc_topicvals in enumerate(curr_besttopicKEYWORD_UNQ_VALUES_perdoc)]
    topickeywords_andvals_perdoc_df = [ [idx, curr_besttopicKEYWORD_UNQ_perdoc[idx], 
                                            curr_besttopicKEYWORD_UNQ_VALUES_perdoc[idx], docs[idx]] for idx in range(numdocs) ]
    topickeywords_andvals_perdoc_df = pd.DataFrame(topickeywords_andvals_perdoc_df, columns = ['doc_id', 'topic_keywords', 'topic_values', 'doc_text']) 
    topickeywords_andvals_perdoc_df.to_csv('RESULT_E-LDA_solution_TOPIC_KEYWORDS_andVALS_perDOCUMENT__'+dataname+'__'+TOPIC_GENERATOR+'.csv', index=False)



    ######  ######  ######  ######  ######   ######
    #         Coherence analysis (optional)       #
    ######  ######  ######  ######  ######   ######
    # DO COHERENCE OF ALL TOPICS IN SOLUTION (not candidate set)
    if DO_MANUAL_COHERENCE:
        T2W_in_solution = T2W[np.unique(solution_set_S[:,1])]

        print('\nComputing coherence of all topics selected by E-LDA solution')
        all_topics_coherence_scores = np.zeros((len(NUM_TOPWORDS_TO_CHECK_COHERENCE_VEC), len(T2W)))
        all_topics_coherence_scores_normalized = np.zeros((len(NUM_TOPWORDS_TO_CHECK_COHERENCE_VEC), len(T2W)))

        doc_term_matrix = D2W.astype(bool)
        # print('\n\nsolution topic mat:\n\n', solution_topic_word_mat,'\n\n\n')
        for checkidx, num_words_intopic_tocheck in enumerate(NUM_TOPWORDS_TO_CHECK_COHERENCE_VEC):
            print(checkidx, num_words_intopic_tocheck)

            topics_coherence_scores_checkidx, all_topics_coherence_scores_normalized_checkidx = \
                                                                        coherence.multitopic_coherence_dense(topic_word_2darray=T2W, 
                                                                        doc_term_matrix=doc_term_matrix,
                                                                        num_topwords_to_check=num_words_intopic_tocheck, 
                                                                        eps_to_add=0.00000000000001)

            all_topics_coherence_scores[checkidx] = topics_coherence_scores_checkidx
            all_topics_coherence_scores_normalized[checkidx] = all_topics_coherence_scores_normalized_checkidx
            print(all_topics_coherence_scores_normalized)


        np.savetxt('RESULT_ALL_generated_topics_coherence_normalized__' + dataname+'__'+TOPIC_GENERATOR, \
              all_topics_coherence_scores_normalized, delimiter=",")

        print('\nMean Coherence of topics IN E-LDA SOLUTION:', \
                    np.mean( all_topics_coherence_scores_normalized[:,np.unique(solution_set_S[:,1])]) )
        print('\nTopic coherence scores saved to file: '+'RESULT_ALL_generated_topics_coherence_normalized__' + dataname+'__'+TOPIC_GENERATOR)


    print('\nE-LDA obtained objective_value:', objective_value)
    print('\nFormatted solution-per-iteration saved to file: ' + 'RESULT_E-LDA_solution_'+dataname+'__'+TOPIC_GENERATOR+'_ANALYZED.csv')
    print('\nTopic keywords and values for each document saved to file: ' + \
            'RESULT_E-LDA_solution_TOPIC_KEYWORDS_andVALS_perDOCUMENT__'+dataname+'__'+TOPIC_GENERATOR+'.csv'+'\n\n')










