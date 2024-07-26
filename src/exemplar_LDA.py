'''
@author: Adam Breuer
'''
import numpy as np
from datetime import datetime # For printing progress and timings
import heapq 



def exemplar_LDA(D2Windexform, T2W, k, time_start=datetime.now()):
    '''Exemplar LDA. This implementation uses indexing (D2Windexform) rather than dot products, which is faster on shorter docs.'''
    '''D2Windexform is a numpy (ragged) array of lists, where the d'th list corresponds to document d, and contains one element per word in document d that is the index of the corresponding word in d.'''
    '''T2W is a dense numpy 2D array of Logged topic-word probabilities (dimensions are numtopics X numwords)'''
    '''k is an int that is the cardinality constraint (TOTAL # doc-topic edges) we seek.'''
    '''time_start is an optional datetime that can be passed to print progress re:the elapsed time including loading/preprocessing the data.'''
    '''exemplar_LDA returns solution_set_S, a 2D numpy array where each row is the [document_id, topic_id] tuple that was added in one iteration of exemplar_LDA.'''

    if not np.all( D2Windexform[0] == np.sort(D2Windexform[0]) ):
        print('\n\nWARNING: word ids not sorted! They should be sorted in each row of D2Windexform, \
                        to accelerate adjacent memory indexing in this exemplar_LDA function!\n')

    # INITIALIZE MEMORY
    solution_set_S = np.zeros((k, 2), dtype=int)
    numdocs = len(D2Windexform)
    curr_docvals = np.log(10**(-100))*np.array([len(doc) for doc in D2Windexform])
    allmargvals_D_x_T = np.zeros((numdocs, T2W.shape[0]))

    time_initialize = datetime.now()
    print('Beginning optimization with fast-start algorithm. Current time:', time_initialize.time(),
        'elapsed time (load data & make topic-word matrix) is', np.round((time_initialize-time_start).total_seconds()/60.,4), 'minutes')

    # COMPUTE FAST START (i.e. add best topic for each document to the solution)
    # This would be much faster with a scipy.sparse dot product...
    for doc_id in range(numdocs):
        allmargvals_D_x_T[doc_id,:] = T2W[:,D2Windexform[doc_id]].sum(axis=1) - curr_docvals[doc_id]
    best_margval_topic_perdoc = np.argmax(allmargvals_D_x_T, 1)
    solution_set_S[:numdocs,0] = range(numdocs)
    solution_set_S[:numdocs,1] = np.argmax(allmargvals_D_x_T, 1)
    curr_docvals = allmargvals_D_x_T[range(numdocs), best_margval_topic_perdoc] + curr_docvals #this has rounding error like 10^-13 due to dot product 

    [T2W[t, D2Windexform[doc_id]] for doc_id, t in enumerate(best_margval_topic_perdoc)]
    curr_bestvals_perword_indoc_DxWidthIndexy = np.array([T2W[t, D2Windexform[doc_id]] \
                                                    for doc_id, t in enumerate(best_margval_topic_perdoc)], dtype=object)
    print('Added best fast-start first topic to each doc, elapsed time is:', 
        np.round((datetime.now()-time_start).total_seconds()/60.,4), 'minutes')

    # RE-INITIALIZE all marginal values
    for doc_id in range((numdocs)):
        allmargvals_D_x_T[doc_id,:] = np.maximum( T2W[:,D2Windexform[doc_id]], \
                                            curr_bestvals_perword_indoc_DxWidthIndexy[doc_id] ).sum(axis=1) - curr_docvals[doc_id]
    best_margval_topic_perdoc_idx = np.argmax(allmargvals_D_x_T, axis=1)
    best_margval_topic_perdoc_value = allmargvals_D_x_T[range(numdocs), best_margval_topic_perdoc_idx]

    time_start_loop = datetime.now()
    print('Completed fast start; beginning main loop; elapsed time is:', 
        np.round((time_start_loop-time_start).total_seconds()/60.,4), 'minutes',
        'fast start time was:', np.round((time_start_loop-time_initialize).total_seconds()/60.,4), 'minutes')

    # MAIN ROUTINE
    # BUILD THE HEAP #
    # HEAPIFY the best_margval_topic_perdoc_values while tracking their doc indices (note that heapq module is a MIN-heap so we negate to get MAX-heap)
    # Each tuple is (negative_margvalue, doc_idx, best_topic_thisdoc_idx). There is one tuple per document.

    heap_best_margval_topic_perdoc_negvalueidxtuples = list(zip(-allmargvals_D_x_T[range(numdocs), \
                                                                best_margval_topic_perdoc_idx], \
                                                                    list(range(numdocs)), best_margval_topic_perdoc_idx))
    allmargvals_D_x_T = None # save memory as this has been heaped
    heapq.heapify( heap_best_margval_topic_perdoc_negvalueidxtuples )
    print('heap initialized incl. allmargvals_D_x_T')

    i = numdocs
    time_start_500iterbatch = datetime.now()
    while i < k:   
        time_start_iter = datetime.now()
        neg_best_margval, best_doc, best_topic = heapq.heappop(heap_best_margval_topic_perdoc_negvalueidxtuples)
        solution_set_S[i] = [best_doc, best_topic]
        i += 1
        curr_bestvals_perword_indoc_DxWidthIndexy[best_doc] = np.maximum( ( T2W[best_topic,:][D2Windexform[best_doc]] ), \
                                                                curr_bestvals_perword_indoc_DxWidthIndexy[best_doc]  ) 
        curr_docvals[best_doc] -= neg_best_margval #note that values were negated in heap because heapq module implements min-heap not max-heap

        allmargvals_D = np.maximum( T2W[:,D2Windexform[best_doc]], \
                            curr_bestvals_perword_indoc_DxWidthIndexy[best_doc] ).sum(axis=1) - curr_docvals[best_doc]
        new_best_topic_for_best_doc = np.argmax(allmargvals_D)
        new_best_doc_tuple = (-allmargvals_D[new_best_topic_for_best_doc], best_doc, new_best_topic_for_best_doc)

        heapq.heappush(heap_best_margval_topic_perdoc_negvalueidxtuples, new_best_doc_tuple)

        if (i % 1000 == 0) and (i != numdocs):
            time_now = datetime.now()
            elapsed_excl_faststart = (time_now-time_start_loop).total_seconds()
            # iter_elapsed = (datetime.now()-time_start_iter).total_seconds()
            print('iteration:', i, 'of k:', k, 'objective value:', np.sum(curr_docvals), \
                'main loop mins:', np.round(elapsed_excl_faststart/60.,4),\
                 ' ~mins rem (based on latest 500 iters):', \
                        np.round((time_now - time_start_500iterbatch).total_seconds() * (k-i)/(60.*500) ,4),\
                 'TOT elapsed incl init:', np.round((time_now-time_start).total_seconds()/60. ,4),\
                 # ' iter:', np.round(iter_elapsed,2), 'sec',\
                 'element added:', [best_doc, best_topic])
                 # ' mins remaining:', np.round((elapsed_excl_faststart * (k-i)/(i-numdocs))/60.,2))
            time_start_500iterbatch = datetime.now()

    print(solution_set_S, '\nOPTIMIZATION COMPLETE.')
    print("TOTAL mins elapsed incl loading data and init:", np.round((datetime.now()-time_start).total_seconds()/60. ,4 ))
    return solution_set_S, np.sum(curr_docvals)



def reconstruct_besttopic_perword_and_objvalues(solution_set_S, D2Windexform, T2W):
    '''Could return these in the exemplar_LDA() algo, but it adds time and mem to compute them, so just reconstruct after we know solution.'''

    time_start = datetime.now()
    k = len(solution_set_S)
    objective_values_perround = np.zeros(k)
    numdocs = len(D2Windexform)

    time_initialize = datetime.now()
    # Due to fast start, first numdocs elements of solution were one-per-doc
    # curr_bestvals_perword_indoc_DxWidthIndexy = np.array([T2W[solution_set_S[ii,1], D2Windexform[solution_set_S[ii,0]]] for ii in range(numdocs)])
    curr_bestvals_perword_indoc_DxWidthIndexy = np.array([T2W[solution_set_S[ii,1], D2Windexform[solution_set_S[ii,0]]] \
                                                    for ii in range(numdocs)], dtype=object)
    curr_docvals = np.asarray( [np.sum(wordvals) for wordvals in curr_bestvals_perword_indoc_DxWidthIndexy] )
    curr_besttopic_perword_indoc_DxWidthIndexy = np.array([np.asarray([solution_set_S[ii,1] \
                                                    for w in D2Windexform[solution_set_S[ii,0]]]) for ii in range(numdocs)], dtype=object)
    objective_values_perround[numdocs-1] = np.sum(curr_docvals)

    for i in range(numdocs, k):   
        best_doc, best_topic = solution_set_S[i] 
        len_best_doc = len(curr_besttopic_perword_indoc_DxWidthIndexy[best_doc])        
        curr_and_new_bestdoc_wordvals = \
            np.vstack(( ( T2W[best_topic,:][D2Windexform[best_doc]] ), curr_bestvals_perword_indoc_DxWidthIndexy[best_doc] ))
        argmax_curr_and_new_bestdoc_wordvals = np.argmax(curr_and_new_bestdoc_wordvals, axis=0)

        curr_bestvals_perword_indoc_DxWidthIndexy[best_doc] = \
            curr_and_new_bestdoc_wordvals[argmax_curr_and_new_bestdoc_wordvals, range(len_best_doc)]

        curr_and_new_bestdoc_wordtopicids = \
            np.vstack(( [best_topic]*len_best_doc, curr_besttopic_perword_indoc_DxWidthIndexy[best_doc] ))
        curr_besttopic_perword_indoc_DxWidthIndexy[best_doc] = \
            curr_and_new_bestdoc_wordtopicids[argmax_curr_and_new_bestdoc_wordvals, range(len_best_doc)]

        curr_docvals[best_doc] = np.sum(curr_bestvals_perword_indoc_DxWidthIndexy[best_doc])
        objective_values_perround[i] = np.sum(curr_docvals)
        # if (i % 100 == 0) and (i != numdocs):
        #     print('iteration:', i, 'of k:', k, 'objective value:', objective_values_perround[i],\
        #         'elapsed:', np.round((datetime.now()-time_start).total_seconds()/60. ,4 ))

    return objective_values_perround, curr_besttopic_perword_indoc_DxWidthIndexy, curr_bestvals_perword_indoc_DxWidthIndexy


